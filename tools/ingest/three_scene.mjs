/** Explicit, source-owned Three.js r186 Scene -> new WebGPU draw execution.
 * The application supplies its pinned Three module, live scene and GPUDevice.
 * No WebGLRenderer is constructed, no scene is cloned, and no frame loop is
 * installed. prepare() is the explicit asynchronous structural-edit boundary;
 * render(camera, attachments) remains synchronous and immediately submits.
 *
 * This admits rigid Mesh/BufferGeometry and the existing material profiles. It
 * is NOT a constructor replacement or complete Three.js renderer compatibility.
 * Unsupported renderable families, shader/render hooks, fog, source environment
 * maps and shadows fail explicitly. Borrowed texture bindings assert completed
 * source uploads; image decoding and texture/sampler realization are not guessed.
 * See THREE_SCENE.md for the supported source and preparation contract.
 */
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuBufferGeometry, bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
export class ThreeSceneError extends Error {
  constructor(code,message){super(`THREE_SCENE_${code}: ${message}`);this.name='ThreeSceneError';this.code='THREE_SCENE_'+code;}
}
const fail=(code,message)=>{throw new ThreeSceneError(code,message);};
const integer=(n,min,max,label)=>{
  if(!Number.isSafeInteger(n)||n<min||n>max)fail('LIMIT',`Invalid ${label}`);return n;
};
const finite=(n,label)=>{if(typeof n!=='number'||!Number.isFinite(n))fail('VALUE',`Invalid ${label}`);return n;};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
const rgba=(color,alpha=1)=>[color.r,color.g,color.b,alpha];
const rgb=color=>[color.r,color.g,color.b];
const position=object=>{const e=object.matrixWorld.elements;return [e[12],e[13],e[14]];};
const MAPS=[['map','baseColorTexture'],['normalMap','normalTexture'],['emissiveMap','emissiveTexture'],
  ['aoMap','occlusionTexture'],['specularMap','specularTexture'],['gradientMap','gradientTexture']];
const DEPTH=['never','always','less','less-equal','equal','greater-equal','greater','not-equal'];

export async function createGpuThreeScene(device,scene,{
  three, textures=new Map(), renderer:renderOptions={}, geometry:geometryOptions={},
  maxNodes=16384,maxGeometries=256,maxBindings=1024,maxGeometryBytes=128*1024*1024,sortObjects=true,
}={}) {
  if(three?.REVISION!=='186'||typeof three.Matrix4!=='function'||typeof three.Frustum!=='function'||
      typeof three.Mesh!=='function'||!(scene instanceof three.Scene))fail('SOURCE','Supply the pinned r186 module and its Scene');
  if(!(textures instanceof Map)||typeof sortObjects!=='boolean')fail('OPTIONS','Expected a texture binding Map and boolean sorting option');
  integer(maxNodes,1,1048576,'node capacity');integer(maxGeometries,1,65536,'geometry capacity');
  integer(maxBindings,1,32768,'material binding capacity');integer(maxGeometryBytes,1,Number.MAX_SAFE_INTEGER,'geometry budget');
  if(renderOptions.format===null||renderOptions.indirectLights===false||renderOptions.threeLights===false||
      (renderOptions.maxMeshes!==undefined&&renderOptions.maxMeshes!==2*maxBindings))
    fail('OPTIONS','Source scenes require color, source light profiles and two preparation slots per binding');
  for(const key of Object.keys(geometryOptions))if(!['maxAttributes','label'].includes(key))fail('OPTIONS',`Unsupported geometry option: ${key}`);
  const models=new Map([
    [three.MeshBasicMaterial.prototype,'unlit'],[three.MeshLambertMaterial.prototype,'lambert'],
    [three.MeshPhongMaterial.prototype,'phong'],[three.MeshToonMaterial.prototype,'toon'],
    [three.MeshStandardMaterial.prototype,'metallic-roughness'],
  ]);
  const geometries=new Map(),materials=new Map();
  let entries=[],lookup=new Map(),renderer,disposed=false,terminal=null,busy=false,preparing=false,prepareVersion=0,sourceDraws=0;
  // End the owner's wait without claiming to cancel already-issued GPU work.
  // Renderer registration still retires its private resources if it resolves late.
  let rejectStopped;
  const stopped=new Promise((_,reject)=>{rejectStopped=reject;});stopped.catch(()=>{});
  const vp=new three.Matrix4(),clip=new three.Matrix4(),frustum=new three.Frustum(),center=new three.Vector3();
  const geometryBytes=()=>[...geometries.values()].reduce((n,g)=>n+g.bufferBytes,0);
  function live(){
    if(disposed)fail('DISPOSED','Source scene bridge is disposed');if(terminal)throw terminal;
    if(renderer?.failed||[...geometries.values()].some(g=>g.failed)){
      terminal=new ThreeSceneError('DEVICE','A source-scene GPU resource failed');release();throw terminal;
    }
  }
  function release(){
    for(const entry of entries)entry.mesh.dispose();entries=[];lookup.clear();
    for(const [m,state] of materials)m.removeEventListener('dispose',state.listener);materials.clear();
    for(const gpu of geometries.values())gpu.dispose();geometries.clear();renderer?.dispose();
  }
  function failed(error){
    if(renderer?.failed||[...geometries.values()].some(g=>g.failed)){terminal??=error;release();}
    throw error;
  }
  function trackMaterial(m){
    if(!materials.has(m)){
      if(materials.size>=2*maxBindings)fail('LIMIT','Source material identity capacity exceeded');
      const state={epoch:0,listener:null};
      state.listener=()=>{state.epoch++;for(const entry of entries)if(entry.material===m)entry.mesh.dispose();};
      materials.set(m,state);m.addEventListener('dispose',state.listener);
    }
    return materials.get(m).epoch;
  }
  function geometryAdmission(g){
    if(!(g instanceof three.BufferGeometry)||g.isInstancedBufferGeometry||Object.values(g.morphAttributes).some(a=>a.length))
      fail('GEOMETRY','This bridge requires rigid BufferGeometry; use the existing skin/morph/instance paths');
    const owners=new Set(Object.values(g.attributes).map(a=>a.isInterleavedBufferAttribute?a.data:a));
    if(g.index)owners.add(g.index);
    if(!Array.isArray(g.groups)||g.groups.length>maxNodes)fail('LIMIT','Geometry group capacity exceeded');
    for(const owner of owners){
      const expected=owner.isInterleavedBuffer?three.InterleavedBuffer.prototype.onUploadCallback:three.BufferAttribute.prototype.onUploadCallback;
      if(owner.onUploadCallback!==expected)fail('HOOK','Effectful attribute upload callbacks require the explicit geometry API');
    }
  }
  function graph(){
    const nodes=[],seen=new Set(),stack=[scene];
    while(stack.length){
      const object=stack.pop();
      if(!(object instanceof three.Object3D)||seen.has(object)||!Array.isArray(object.children))fail('GRAPH','Expected an acyclic source hierarchy');
      seen.add(object);nodes.push(object);if(nodes.length>maxNodes)fail('LIMIT','Source node capacity exceeded');
      if(object.children.length+stack.length+nodes.length>maxNodes)fail('LIMIT','Source node capacity exceeded');
      if(object.onBeforeRender!==three.Object3D.prototype.onBeforeRender||object.onAfterRender!==three.Object3D.prototype.onAfterRender)
        fail('HOOK','Custom render callbacks are not admitted by this source bridge');
      if(object.isMesh){
        if(Array.isArray(object.material)&&object.material.length>maxBindings)fail('LIMIT','Source material array exceeds capacity');
        if(object.isSkinnedMesh||object.isInstancedMesh||object.isBatchedMesh||object.intersectsFrustum!==three.Mesh.prototype.intersectsFrustum)
          fail('OBJECT','Use the explicit animation/instance path for this mesh family');
        if(object.castShadow||object.receiveShadow)fail('SHADOW','Source shadow ownership is not inferred; use the explicit scene shadow API');
        geometryAdmission(object.geometry);
      } else if(object.isLine||object.isPoints||object.isSprite||object.isLightProbe||object.isLightProbeGrid)
        fail('OBJECT',`Unsupported source renderable: ${object.type}`);
      if(object.isLight)light(object);
      for(let i=object.children.length-1;i>=0;i--)stack.push(object.children[i]);
    }
    if(scene.fog!==null||scene.environment!==null||(scene.background!==null&&!scene.background.isColor))
      fail('SCENE','Fog, source environment maps and texture backgrounds require their own rendering paths');
    return nodes;
  }
  function light(source){
    if(source.castShadow)fail('SHADOW','A source shadow light cannot silently become an unshadowed light');
    let type;
    if(source.isAmbientLight)type='ambient';else if(source.isHemisphereLight)type='hemisphere';
    else if(source.isDirectionalLight)type='directional';else if(source.isPointLight)type='point';
    else if(source.isSpotLight)type='spot';else fail('LIGHT',`Unsupported source light: ${source.type}`);
    if(source.map)fail('LIGHT','Projected source light textures are not admitted');
    const result={type,color:rgb(source.color),intensity:source.intensity};
    if(result.color.some(v=>typeof v!=='number'||!Number.isFinite(v)||v<0||v>1)||
        finite(source.intensity,'light intensity')<0||!Number.isFinite(Math.fround(source.intensity)))fail('LIGHT','Invalid source radiance');
    if(type==='hemisphere'){result.direction=position(source);result.groundColor=rgb(source.groundColor);}
    if(type==='point'||type==='spot'){
      result.position=position(source);result.decay=source.decay;
      if(finite(source.decay,'light decay')<0||finite(source.distance,'light distance')<0)fail('LIGHT','Invalid source light falloff');
      if(source.distance!==0)result.range=source.distance;
    }
    if(type==='directional'||type==='spot'){
      const a=position(source),b=position(source.target);result.direction=b.map((v,i)=>v-a[i]);
    }
    if(type==='spot'){
      if(finite(source.angle,'spot angle')<0||source.angle>Math.PI/2||finite(source.penumbra,'spot penumbra')<0||source.penumbra>1)
        fail('LIGHT','Invalid source spot cone');
      result.outerConeAngle=source.angle;result.innerConeAngle=source.angle*(1-source.penumbra);
    }
    return result;
  }
  function materialDescription(m){
    const shading=models.get(Object.getPrototypeOf(m));
    if(!shading)fail('MATERIAL',`Unsupported source material: ${m?.type}`);
    for(const descriptor of Object.values(Object.getOwnPropertyDescriptors(m)))
      if(!Object.hasOwn(descriptor,'value'))fail('HOOK','Accessor-backed material fields are not admitted');
    if(m.onBeforeRender!==three.Material.prototype.onBeforeRender||m.onBeforeCompile!==three.Material.prototype.onBeforeCompile||
        m.customProgramCacheKey!==three.Material.prototype.customProgramCacheKey)fail('HOOK','Custom material shader/render hooks require their original component');
    if(m.wireframe||m.alphaHash||m.alphaToCoverage||m.premultipliedAlpha||m.stencilWrite||m.polygonOffset||m.clippingPlanes?.length)
      fail('MATERIAL','Wireframe, hashed/coverage alpha, premultiplication, stencil, polygon offset and clipping are not admitted');
    if(m.blending!==three.NormalBlending&&!(m.blending===three.NoBlending&&!m.transparent))fail('MATERIAL','Unsupported source blending mode');
    if(m.transparent&&m.alphaTest>0)fail('MATERIAL','Combined transparent alpha testing requires an extended material profile');
    for(const key of ['lightMap','bumpMap','displacementMap','alphaMap','envMap'])if(m[key])fail('MATERIAL',`Unsupported source map: ${key}`);
    if(![0,1,2].includes(m.side)||!Number.isInteger(m.depthFunc)||!DEPTH[m.depthFunc])fail('MATERIAL','Unsupported side/depth state');
    if(finite(m.alphaTest,'alpha test')<0||m.alphaTest>1)fail('MATERIAL','Invalid source alpha test');
    for(const key of ['transparent','vertexColors','depthTest','depthWrite','colorWrite','forceSinglePass'])
      if(typeof m[key]!=='boolean')fail('MATERIAL',`Expected boolean ${key}`);
    const options={shading,vertexColors:m.vertexColors,flatShading:shading==='unlit'?false:m.flatShading===true,
      alphaMode:m.transparent?'BLEND':m.alphaTest>0?'MASK':'OPAQUE',alphaCutoff:m.alphaTest>0?m.alphaTest:0.5,
      depthTest:m.depthTest,depthWrite:m.depthWrite,depthCompare:DEPTH[m.depthFunc],colorWrite:m.colorWrite};
    const values={baseColor:rgba(m.color,m.opacity)};
    if(options.alphaMode==='MASK')values.alphaCutoff=m.alphaTest;
    if(shading!=='unlit')values.emissiveFactor=rgb(m.emissive).map(v=>v*m.emissiveIntensity);
    if(shading==='phong'){values.specularColor=rgb(m.specular);values.shininess=m.shininess;}
    if(shading==='metallic-roughness'){values.metallicFactor=m.metalness;values.roughnessFactor=m.roughness;}
    let transform=null;const textureKey=[];
    function texture(t,field){
      if(!(t instanceof three.Texture)||t.isCubeTexture||t.isVideoTexture||t.channel!==0)fail('TEXTURE','Expected a current, ordinary UV0 texture binding');
      const binding=textures.get(t);
      if(!binding?.view||!binding.sampler||binding.version!==t.version||binding.sourceVersion!==t.source.version)
        fail('TEXTURE','Supply a completed texture binding matching both texture and source upload versions');
      options[field]={view:binding.view,sampler:binding.sampler};textureKey.push(field,t,binding.view,binding.sampler);
      if(field!=='gradientTexture'){
        if(t.matrixAutoUpdate)t.updateMatrix();const e=t.matrix.elements,next=[e[0],e[1],e[3],e[4],e[6],e[7]];
        if(transform&&!same(transform,next))fail('TEXTURE','Independent mutable map transforms require an extended geometry profile');
        transform=next;
      }
    }
    for(const [source,field] of MAPS)if(m[source])texture(m[source],field);
    if(m.metalnessMap||m.roughnessMap){
      if(m.metalnessMap!==m.roughnessMap)fail('TEXTURE','Separate metallic and roughness maps require independent shader bindings');
      texture(m.metalnessMap,'metallicRoughnessTexture');
    }
    if(m.normalMap){
      if(m.normalMapType!==three.TangentSpaceNormalMap||m.normalScale.x!==m.normalScale.y)fail('MATERIAL','The current profile requires tangent-space maps and equal XY normal scale');
      values.normalScale=m.normalScale.x;
    }
    if(m.aoMap)values.occlusionStrength=m.aoMapIntensity;
    if(transform)values.uvTransform=transform;
    for(const [key,value] of Object.entries(values))for(const word of Array.isArray(value)?value:[value])
      if(!Number.isFinite(Math.fround(finite(word,key))))fail('VALUE',`${key} exceeds the material f32 profile`);
    if(values.baseColor.some(v=>v<0)||m.opacity>1||(shading!=='unlit'&&values.baseColor.some(v=>v>1))||
        values.emissiveFactor?.some(v=>v<0)||values.specularColor?.some(v=>v<0)||values.shininess<0||
        values.metallicFactor<0||values.metallicFactor>1||values.roughnessFactor<0||values.roughnessFactor>1||
        values.occlusionStrength<0||values.occlusionStrength>1)fail('VALUE','Source material is outside the admitted factor profile');
    const epoch=trackMaterial(m);
    const sides=m.transparent&&m.side===three.DoubleSide&&!m.forceSinglePass?['back','front']:[['front','back','double'][m.side]];
    return sides.map(side=>{
      const config={...options,side};
      const structural=[epoch,shading,side,config.vertexColors,config.flatShading,config.alphaMode,
        config.depthTest,config.depthWrite,config.depthCompare,config.colorWrite,...textureKey];
      return {options:config,values,structural};
    });
  }
  function desired(nodes){
    const out=[],descriptions=new Map(),seen=new Map();
    const get=m=>{if(!descriptions.has(m))descriptions.set(m,materialDescription(m));return descriptions.get(m);};
    if(scene.overrideMaterial)get(scene.overrideMaterial);
    for(const object of nodes)if(object.isMesh){
      const g=object.geometry;
      const source=Array.isArray(object.material)?object.material:[object.material];
      for(const original of source){
        if(!original)continue;
        // The original controls visibility/list admission even with an override.
        get(original);
        const m=scene.overrideMaterial&&original.allowOverride===true?scene.overrideMaterial:original;
        let set=seen.get(g);if(!set)seen.set(g,set=new Set());if(set.has(m))continue;set.add(m);
        for(const desc of get(m)){
          out.push({geometry:g,material:m,desc});
          if(out.length>maxBindings||seen.size>maxGeometries)fail('LIMIT','Source geometry/material binding capacity exceeded');
        }
      }
    }
    if(seen.size>maxGeometries||out.length>maxBindings)fail('LIMIT','Source geometry/material binding capacity exceeded');
    return out;
  }
  function sameDesired(a,b){return a.length===b.length&&a.every((item,i)=>item.geometry===b[i].geometry&&item.material===b[i].material&&same(item.desc.structural,b[i].desc.structural));}
  function publish(next){
    for(const entry of entries)if(!next.includes(entry))entry.mesh.dispose();
    entries=next;lookup=new Map();const used=new Set(next.map(e=>e.geometry)),usedMaterials=new Set();
    for(const entry of next){
      let byMaterial=lookup.get(entry.geometry);if(!byMaterial)lookup.set(entry.geometry,byMaterial=new Map());
      let records=byMaterial.get(entry.material);if(!records)byMaterial.set(entry.material,records=[]);records.push(entry);usedMaterials.add(entry.material);
    }
    for(const [g,gpu] of geometries)if(!used.has(g)){gpu.dispose();geometries.delete(g);}
    for(const [m,state] of materials)if(!usedMaterials.has(m)){m.removeEventListener('dispose',state.listener);materials.delete(m);}
  }
  async function prepare(){
    live();if(busy)fail('REENTRANT','A source-scene operation is already running');busy=true;preparing=true;
    const created=[],added=[];
    try{
      const request=desired(graph()),next=[];
      for(const item of request){
        live();let gpu=geometries.get(item.geometry);
        if(!gpu){
          gpu=createGpuBufferGeometry(device,item.geometry,{...geometryOptions,maxBytes:maxGeometryBytes,
            maxInitialBytes:Math.max(0,maxGeometryBytes-geometryBytes())});
          geometries.set(item.geometry,gpu);added.push(item.geometry);
        }else gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});
        const signature=bufferGeometrySnapshot(gpu,device).signature;
        let entry=lookup.get(item.geometry)?.get(item.material)?.find(e=>!e.mesh.disposed&&e.signature===signature&&same(e.structural,item.desc.structural));
        if(!entry){
          const mesh=await Promise.race([renderer.addMesh(gpu,{...item.desc.options,...item.desc.values}),stopped]);
          entry={geometry:item.geometry,material:item.material,structural:item.desc.structural,signature,mesh};created.push(entry);live();
        }
        next.push(entry);
      }
      // A layout can change while a pipeline await is outstanding, even when
      // the source geometry identity is unchanged. Do not publish that stale
      // registration; current content versions are uploaded at this boundary.
      const checked=new Set();
      for(const entry of next){
        const gpu=geometries.get(entry.geometry);
        if(!checked.has(gpu)){gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});checked.add(gpu);}
        if(bufferGeometrySnapshot(gpu,device).signature!==entry.signature)fail('CHANGED','Geometry layout changed during preparation');
      }
      if(!sameDesired(request,desired(graph())))fail('CHANGED','Source structure changed while pipelines were being prepared');
      publish(next);prepareVersion++;return bridge;
    }catch(error){
      for(const entry of created)entry.mesh.dispose();
      for(const g of added){geometries.get(g)?.dispose();geometries.delete(g);}
      return failed(error);
    }finally{busy=false;preparing=false;}
  }
  function cameraFrame(camera){
    if(!(camera instanceof three.Camera)||(!camera.isPerspectiveCamera&&!camera.isOrthographicCamera)||camera.isArrayCamera||camera.reversedDepth)
      fail('CAMERA','Supply one non-reversed perspective or orthographic source camera');
    if(![three.WebGLCoordinateSystem,three.WebGPUCoordinateSystem].includes(camera.coordinateSystem))fail('CAMERA','Unknown source clip convention');
    if(scene.matrixWorldAutoUpdate===true)scene.updateMatrixWorld();
    if(camera.parent===null&&camera.matrixWorldAutoUpdate===true)camera.updateMatrixWorld();
    vp.multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(vp,camera.coordinateSystem,false);clip.copy(vp);
    if(camera.coordinateSystem===three.WebGLCoordinateSystem)
      for(let c=0;c<4;c++)clip.elements[c*4+2]=0.5*(vp.elements[c*4+2]+vp.elements[c*4+3]);
    return camera.isOrthographicCamera?{viewDirection:[camera.matrixWorld.elements[8],camera.matrixWorld.elements[9],camera.matrixWorld.elements[10]]}:
      {cameraPosition:position(camera)};
  }
  function render(camera,frame){
    live();if(busy)fail('REENTRANT','A source-scene operation is already running');busy=true;
    try{
      if(!frame||typeof frame!=='object')fail('FRAME','Supply borrowed render attachments');
      for(const key of ['draws','viewProjection','lighting'])if(Object.hasOwn(frame,key))fail('FRAME',`${key} belongs to the source scene/camera`);
      graph();const lighting=cameraFrame(camera);lighting.lights=[];
      const opaque=[],transparent=[],stack=[{object:scene,groupOrder:0}],descriptions=new Map();
      const get=m=>{if(!descriptions.has(m))descriptions.set(m,materialDescription(m));return descriptions.get(m);};
      let visited=0;
      while(stack.length){
        const item=stack.pop(),object=item.object;let groupOrder=item.groupOrder;
        if(++visited>maxNodes)fail('LIMIT','Source traversal capacity exceeded');
        if(object.visible===false)continue;
        if(object.layers.test(camera.layers)){
          if(object.isGroup)groupOrder=object.renderOrder;
          else if(object.isLOD){if(object.autoUpdate)object.update(camera);}
          else if(object.isLight)lighting.lights.push(light(object));
          else if(object.isMesh&&(!object.frustumCulled||object.intersectsFrustum(frustum))){
            const g=object.geometry;
            let z=0;
            if(sortObjects){
              const bounds=object.boundingSphere!==undefined?object:g;
              if(bounds.boundingSphere===null)bounds.computeBoundingSphere();
              z=center.copy(bounds.boundingSphere.center).applyMatrix4(object.matrixWorld).applyMatrix4(vp).z;
            }
            function push(original,group){
              if(!original||!original.visible)return;
              const material=scene.overrideMaterial&&original.allowOverride===true?scene.overrideMaterial:original;
              const desc=get(material),records=lookup.get(g)?.get(material);
              const bindings=desc.map(d=>records?.find(e=>!e.mesh.disposed&&same(e.structural,d.structural)));
              if(bindings.some(e=>!e))fail('PREPARE','Call prepare() after changing geometry, material structure or texture bindings');
              if(opaque.length+transparent.length>=(renderOptions.maxDraws??1024))fail('LIMIT','Source draw list exceeds capacity');
              // Source list partition and sorting precede the draw-time override.
              (original.transparent?transparent:opaque).push({object,geometry:g,material,listMaterial:original,group,groupOrder,z,desc,bindings});
            }
            if(Array.isArray(object.material))for(const group of g.groups)push(object.material[group.materialIndex],group);
            else push(object.material,null);
          }
        }
        for(let i=object.children.length-1;i>=0;i--)stack.push({object:object.children[i],groupOrder});
      }
      if(lighting.lights.length>8)fail('LIMIT','Visible source lights exceed the renderer capacity');
      if(sortObjects){
        const order=(a,b)=>a.groupOrder-b.groupOrder||a.object.renderOrder-b.object.renderOrder;
        opaque.sort((a,b)=>order(a,b)||a.listMaterial.id-b.listMaterial.id||a.z-b.z||a.object.id-b.object.id);
        transparent.sort((a,b)=>order(a,b)||b.z-a.z||a.object.id-b.object.id);
      }
      const items=[...opaque,...transparent],draws=[];
      if(items.reduce((n,item)=>n+item.bindings.length,0)>(renderOptions.maxDraws??1024))fail('LIMIT','Expanded source draws exceed capacity');
      const updated=new Set();
      for(const item of items){
        const gpu=geometries.get(item.geometry);
        if(!updated.has(gpu)){gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});updated.add(gpu);}
        const shape=bufferGeometrySnapshot(gpu,device);
        if(item.bindings.some(e=>e.signature!==shape.signature))fail('PREPARE','Geometry layout changed; call prepare() before drawing it');
        const extent=shape.indexBuffer?shape.indexCount:shape.vertexCount;
        const start=item.group?integer(item.group.start,0,Number.MAX_SAFE_INTEGER,'group start'):0;
        const length=item.group?.count??Infinity;
        if(length!==Infinity)integer(length,0,Number.MAX_SAFE_INTEGER,'group count');
        const first=Math.min(start,extent),count=Math.min(length,extent-first);
        item.object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse,item.object.matrixWorld);
        item.object.normalMatrix.getNormalMatrix(item.object.modelViewMatrix);
        for(let i=0;i<item.bindings.length;i++)draws.push({mesh:item.bindings[i].mesh,
          worldMatrix:item.object.matrixWorld.elements,first,count,...item.desc[i].values});
      }
      // The retained renderer updates these public versions twice per double-
      // sided transparent item. No source callback is erased or replayed here:
      // custom callbacks were rejected before any frame work.
      for(const item of items)if(item.bindings.length===2){
        item.material.side=three.BackSide;item.material.needsUpdate=true;
        item.material.side=three.FrontSide;item.material.needsUpdate=true;
        item.material.side=three.DoubleSide;
      }
      const prepared={...frame,viewProjection:clip.elements,lighting,draws};
      if(scene.background!==null){prepared.clearColor=rgba(scene.background);prepared.loadOp='clear';}
      renderer.render(prepared);sourceDraws=items.length;return bridge;
    }catch(error){return failed(error);}finally{busy=false;}
  }
  const bridge=Object.freeze({scene,prepare,render,
    get disposed(){return disposed;},get failed(){return terminal!==null||!!renderer?.failed||[...geometries.values()].some(g=>g.failed);},
    get diagnostics(){return Object.freeze({prepareVersion,sourceDraws,logicalDraws:renderer?.drawCount??0,
      drawCalls:renderer?.drawCallCount??0,geometryCount:geometries.size,geometryBytes:geometryBytes(),
      materialBindings:entries.filter(e=>!e.mesh.disposed).length,rendererBytes:renderer?.allocatedBytes??0,
      bundles:renderer?.bundleDiagnostics??null});},
    async whenIdle(){live();try{await Promise.race([Promise.all([renderer.whenIdle(),...[...geometries.values()].map(g=>g.whenIdle())]),stopped]);live();return bridge;}catch(error){return failed(error);}},
    dispose(){if(busy&&!preparing)fail('REENTRANT','Cannot dispose during source submission');if(!disposed){disposed=true;rejectStopped(new ThreeSceneError('DISPOSED','Source scene bridge is disposed'));release();}},
  });
  try{
    // Source validation precedes even the renderer's uniform allocation.
    desired(graph());
    renderer=await createGpuAnimationRenderer(device,{...renderOptions,indirectLights:true,threeLights:true,maxMeshes:2*maxBindings});
    live();await prepare();return bridge;
  }catch(error){disposed=true;release();throw error;}
}
