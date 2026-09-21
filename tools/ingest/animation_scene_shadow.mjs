/** Scene-owned shadow orchestration. Prepared from the scene's material snapshots;
 * shares its deformers, never advances or owns its pose. Bounds include receivers
 * and off-camera casters. GPU map and CPU summary budgets are separate from the
 * color/deformation budget. Loaded only by shadow-enabled scene construction.
 */
import {createAnimationBounds} from './animation_bounds.mjs';
import {createGpuAnimationShadowMap} from './animation_shadow.mjs';
import {animationShadowWorldBounds,fitAnimationShadowView} from './animation_shadow_view.mjs';
const fail = message => { const error = new Error(`ANIMATION_SCENE_SHADOW: ${message}`); error.code = 'ANIMATION_SCENE_SHADOW'; throw error; };
const positive = (v,name) => { if(!Number.isSafeInteger(v)||v<1)fail(`Invalid ${name}`); return v; };
function array(value, length, name) {
  if ((!Array.isArray(value)&&!ArrayBuffer.isView(value))||value.length!==length)fail(`Invalid ${name}`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer)||value.buffer.resizable)fail(`${name} needs fixed unshared storage`);
    try {new Uint8Array(value.buffer,0,0);}catch{fail(`${name} is detached`);}
  }
  return Array.from(value);
}
function snapshotLighting(input) {
  if (!input||typeof input!=='object'||Array.isArray(input))fail('Automatic shadows require frame lighting');
  const copy={...input};
  for(const key of ['cameraPosition','viewDirection'])if(copy[key]!==undefined)copy[key]=array(copy[key],3,key);
  if(!Array.isArray(copy.lights)||copy.lights.length>8)fail('Expected up to eight frame lights');
  copy.lights=copy.lights.map(source=>{
    if(!source||typeof source!=='object'||Array.isArray(source))fail('Invalid frame light');
    const light={...source};
    for(const key of ['position','direction','color'])if(light[key]!==undefined)light[key]=array(light[key],3,key);
    return light;
  });
  // Unknown keys remain intact for the existing color renderer to reject.
  return copy;
}

/** Internal two-phase construction: validate/scan without GPU work, then bind
 * the already-created scene deformers. No duplicate geometry deformation.
 */
export function prepareAnimationSceneShadows(pose, inputs, options) {
  if(!options||typeof options!=='object'||Array.isArray(options))fail('Expected shadow options');
  const allowed=['lightIndex','width','height','maxBytes','maxBoundsBytes','maxBoundsComponents',
    'padding','minNear','bias','normalBias','strength','casters','blend','label'];
  for(const key of Object.keys(options))if(!allowed.includes(key))fail(`Unsupported shadow option: ${key}`);
  const {lightIndex=0,width=1024,height=width,maxBytes=64*1024*1024,maxBoundsBytes=16*1024*1024,
    maxBoundsComponents=16777216,padding=0.05,minNear=0.001,bias=0.0005,normalBias=0,strength=1,
    blend='reject',label='f3d-scene-shadow'}=options;
  if(!Number.isInteger(lightIndex)||lightIndex<0||lightIndex>7)fail('lightIndex must be in 0..7');
  for(const [name,value]of Object.entries({width,height,maxBytes,maxBoundsBytes,maxBoundsComponents}))positive(value,name);
  if(!Number.isSafeInteger(width*height*4)||width*height*4>=maxBytes)fail('Shadow texture exceeds its GPU budget');
  if(typeof label!=='string'||!['reject','skip'].includes(blend))fail('Invalid label or blend policy');
  if([bias,normalBias,strength].some(v=>typeof v!=='number'||!Number.isFinite(Math.fround(v)))||Math.abs(bias)>1||normalBias<0||strength<0||strength>1)fail('Invalid shadow bias or strength');
  // Exercise exactly the same fit-option admission before allocating a map.
  fitAnimationShadowView({type:'directional'},{min:[0,0,0],max:[1,1,1]},{padding,minNear});
  if(!Array.isArray(inputs)||!inputs.length||inputs.length>4096)fail('Expected 1..4096 scene meshes');
  const selected=options.casters===undefined?inputs.map(({material})=>material.alphaMode!=='BLEND'||blend!=='skip'):
    array(options.casters,inputs.length,'caster selection');
  if(selected.some(v=>typeof v!=='boolean'))fail('Caster selection must contain booleans');
  const meshCount=inputs.length;
  let materials=inputs.map(({material},i)=>{
    if(!selected[i])return null;
    if(material.alphaMode==='BLEND')fail('BLEND has no depth-only shadow policy; select blend:"skip" or exclude it in casters');
    const caster={};
    for(const key of ['indices','baseColor','doubleSided','alphaMode','alphaCutoff','texCoords','vertexColors','baseColorTexture','uvTransform']) {
      if(material[key]!==undefined)caster[key]=material[key];
    }
    // The color scene already snapshotted these arrays. Other lit maps and their
    // UV sets cannot affect shadow alpha and must not reach the unlit caster.
    if(material.mapCoordinates?.baseColorTexture)caster.mapCoordinates={baseColorTexture:material.mapCoordinates.baseColorTexture};
    return caster;
  });
  const initialVersion=pose.version,summaries=[];
  let summaryBytes=0,components=0,map=null,deformers=null,handles=[],disposed=false,initializing=false,ready=false;
  function release(){map?.dispose();for(const summary of summaries)summary.dispose();handles=[];deformers=null;}
  function live(){if(disposed||pose.disposed)fail('Scene shadows or pose have been disposed');}
  try {
    for(const {geometry}of inputs) {
      const summary=createAnimationBounds(pose,geometry,{maxBytes:positive(maxBoundsBytes-summaryBytes,'remaining bounds bytes'),
        maxComponents:positive(maxBoundsComponents-components,'remaining bounds components')});
      summaries.push(summary);summaryBytes+=summary.byteLength;components+=summary.sourceComponents;
    }
    if(pose.version!==initialVersion)fail('Pose changed while preparing shadow bounds');
  }catch(error){release();throw error;}
  inputs=null;
  const state=Object.freeze({
    get allocatedBytes(){return map?.allocatedBytes??0;},get boundsBytes(){return disposed?0:summaryBytes;},
    get failed(){return map?.failed??false;},
    async initialize(device, values){
      live();if(initializing||ready)fail('Scene shadows have already been initialized');initializing=true;
      try {
        if(!Array.isArray(values)||values.length!==meshCount)fail('Mismatched scene deformers');
        deformers=values.slice();
        if(pose.version!==initialVersion||deformers.some(g=>g.poseVersion!==initialVersion||g.disposed||g.failed))fail('Pose changed before shadow binding');
        map=await createGpuAnimationShadowMap(device,{width,height,maxBytes,maxDraws:Math.max(1,materials.filter(Boolean).length),
          maxMeshes:Math.max(1,materials.filter(Boolean).length),label});
        live();
        for(let i=0;i<materials.length;i++)if(materials[i]) {
          handles[i]=await map.addMesh(deformers[i],materials[i]);live();
        }
        if(pose.version!==initialVersion||deformers.some(g=>g.poseVersion!==initialVersion||g.disposed||g.failed))fail('Pose changed during shadow binding');
        materials=[];ready=true;return state;
      }catch(error){disposed=true;release();throw error;}
      finally{initializing=false;}
    },
    // Select before fitting; off-camera selected casters still cast shadows.
    // Color-frustum visibility is intentionally NOT the caster selection.
    render(inputLighting,drawIndices=null){
      live();if(!ready)fail('Initialize the scene shadow map before rendering');
      let indices=Array.from({length:meshCount},(_,i)=>i);
      if(drawIndices!==null) {
        if(!Array.isArray(drawIndices)||!drawIndices.length||drawIndices.length>meshCount)fail('Invalid active draw selection');
        const selected=new Set();
        for(const index of drawIndices) {
          if(!Number.isSafeInteger(index)||index<0||index>=meshCount||selected.has(index))fail('Invalid or duplicate active draw index');
          selected.add(index);
        }
        indices=indices.filter(index=>selected.has(index));
      }
      const activeHandles=indices.map(i=>handles[i]).filter(Boolean);
      const version=pose.version;
      const lighting=snapshotLighting(inputLighting),light=lighting.lights[lightIndex];
      if(!light)fail('Selected lightIndex is absent from frame lighting');
      const versions=deformers.map(g=>g.version);
      const world=animationShadowWorldBounds(indices.map(i=>({
        bounds:summaries[i].snapshot.poseVersion===version?summaries[i].snapshot:summaries[i].update(),worldMatrix:deformers[i].worldMatrix,
      })));
      const view=fitAnimationShadowView(light,world,{padding,minNear});
      live();
      if(pose.version!==version||deformers.some((g,i)=>g.poseVersion!==version||g.version!==versions[i]||g.disposed||g.failed))fail('Upload a stable complete pose before automatic shadows');
      // One depth submission always precedes the receiving color submission.
      // Re-render even at the same pose: source alpha textures can change without
      // advancing pose.version, and lighting may change independently as well.
      map.render({viewProjection:view.viewProjection,draws:activeHandles});
      return {lighting,shadow:{map,lightIndex,bias,normalBias,strength},stats:Object.freeze({
        poseVersion:version,lightIndex,casterCount:activeHandles.length,mapVersion:map.version,view,
      })};
    },
    async whenIdle(){live();if(map)await map.whenIdle();live();},
    dispose(){if(!disposed){disposed=true;release();}},
  });
  return state;
}
