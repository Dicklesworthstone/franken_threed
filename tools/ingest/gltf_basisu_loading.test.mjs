import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

// Production asset I/O, model/material preparation, texture management, KTX2
// validation and retained-callback adapter run together. Geometry/pose decoding,
// meshopt/Draco (unused by these fixtures), the final GPU scene, native image
// decoding and KTX2 decoder output are explicit doubles. No Basis bitstream or
// native GPU execution is claimed. Use --experimental-vm-modules to see all cases.
if (!process.execArgv.includes('--experimental-vm-modules')) {
  test('BasisU asset-to-owning-scene integration regressions',()=>{
    // Do not inherit the parent runner's private IPC/reporting context.
    const env={...process.env};delete env.NODE_TEST_CONTEXT;
    const out=execFileSync(process.execPath,['--experimental-vm-modules','--test','--test-reporter=tap',fileURLToPath(import.meta.url)],
      {env,encoding:'utf8',timeout:20000,stdio:['ignore','pipe','pipe']});
    assert.match(out,/# fail 0/);assert.match(out,/# pass [1-9]\d*/);
  });
} else {
  const vm=await import('node:vm');
  const {createGltfTextureResources,GltfTextureError}=await import('./gltf_textures.mjs');
  const {inspectGltfKtx2}=await import('./gltf_ktx2.mjs');
  const source={};
  for(const name of ['gltf_asset','animation_model','gltf_scene_loader'])source[name]=await readFile(new URL('./'+name+'.mjs',import.meta.url),'utf8');
  const encode=json=>new TextEncoder().encode(JSON.stringify(json));
  const uri=(data,mime)=>'data:'+mime+';base64,'+Buffer.from(data).toString('base64');
  const png=Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==','base64'));
  const extension='KHR_texture_basisu',origin='https://asset.test/';
  // Structurally valid UASTC headers/ranges with SYNTHETIC payloads, not a codec
  // fixture. Each 4x4, 2x2 and 1x1 level occupies one encoded UASTC block.
  function ktx2({linear=false,levels=3}={}) {
    const dfd=80+24*levels,start=Math.ceil((dfd+44)/16)*16;
    const bytes=new Uint8Array(start+16*levels),v=new DataView(bytes.buffer);
    bytes.set([171,75,84,88,32,50,48,187,13,10,26,10]);
    for(const [offset,value]of [[16,1],[20,4],[24,4],[36,1],[40,levels],[48,dfd],[52,44],[dfd,44]])v.setUint32(offset,value,true);
    v.setUint16(dfd+8,2,true);v.setUint16(dfd+10,40,true);
    bytes.set([166,linear?0:1,linear?1:2,0,3,3,0,0,16],dfd+12);
    for(let i=0;i<levels;i++) {
      v.setBigUint64(80+i*24,BigInt(start+i*16),true);v.setBigUint64(88+i*24,16n,true);v.setBigUint64(96+i*24,16n,true);
      bytes.fill(20+i,start+i*16,start+(i+1)*16);
    }
    return bytes;
  }
  function glb(json,bin) {
    const text=encode(json),length=Math.ceil(text.length/4)*4,binLength=Math.ceil(bin.length/4)*4;
    const out=new Uint8Array(28+length+binLength),v=new DataView(out.buffer);
    for(const [o,n]of [[0,0x46546c67],[4,2],[8,out.length],[12,length],[16,0x4e4f534a],[20+length,binLength],[24+length,0x004e4942]])v.setUint32(o,n,true);
    out.fill(32,20,20+length);out.set(text,20);out.set(bin,28+length);return out;
  }
  function fixture(required=false) {
    return {asset:{version:'2.0'},extensionsUsed:[extension],...(required?{extensionsRequired:[extension]}:{}),
      materials:[{extensions:{KHR_materials_unlit:{}},pbrMetallicRoughness:{baseColorTexture:{index:0}}}],
      textures:[{...(required?{}:{source:0}),extensions:{[extension]:{source:1}}}],
      images:[{uri:'fallback.png'},{uri:'color.ktx2'}],
    };
  }
  class BoundaryError extends Error {constructor(code,message){super(message);this.code=code;}}
  const unused=()=>assert.fail('Unexpected boundary call');
  async function link(name,exports) {
    const module=new vm.SourceTextModule(source[name]);
    await module.link(specifier=>{
      const values=exports[specifier];assert.ok(values,'Unexpected dependency: '+specifier);
      return new vm.SyntheticModule(Object.keys(values),function(){for(const [k,v]of Object.entries(values))this.setExport(k,v);});
    });await module.evaluate();return module.namespace;
  }
  // Identity codec plans are valid only for our uncompressed-geometry fixtures.
  const codec=json=>({skippedBuffers:[],decodedBytes:0,decode:async buffers=>({json,buffers,decodedBytes:0,decodedBufferViews:0,decodedPrimitives:0})});
  const asset=await link('gltf_asset',{
    './gltf_meshopt.mjs':{prepareMeshoptBuffers:codec},'./gltf_draco.mjs':{prepareDracoMeshes:codec},
  });
  const model=await link('animation_model',{
    './animation_gltf.mjs':{decodeGltfAnimation:()=>({nodes:[]})},
    './animation_geometry.mjs':{decodeGltfGeometry:json=>({scene:0,diagnostics:[],primitives:json.materials.map((_,material)=>({
      node:material,mesh:0,primitive:material,material,geometry:{positions:Float64Array.of(0,0,0,1,0,0,0,1,0)},
      indices:Uint32Array.of(0,1,2),attributes:{TEXCOORD_0:{values:Float64Array.of(0,0,1,0,0,1)}},
    }))})},
    './animation_runtime.mjs':{AnimationPoseError:BoundaryError,createAnimationPlayer:unused},
    './animation_deformer.mjs':{createAnimationDeformer:unused},
    './animation_model_export.mjs':{createAnimationModelExporter:unused,AnimationExportError:BoundaryError},
    './animation_model_pick.mjs':{createAnimationModelPicker:unused,AnimationRaycastError:BoundaryError},
    './gltf_scene_view.mjs':{decodeGltfSceneView:()=>({cameras:[],lights:[]}),createGltfSceneView:unused,GltfSceneViewError:BoundaryError},
  });
  async function harness({format=36492,features=['texture-compression-bc'],late=false,sceneError=null,writeErrorAt=-1,queueError=null}={}) {
    const log={fetch:[],textures:[],writes:[],samplers:[],parses:0,temporaryDisposals:0,bitmapCloses:0,models:[],copies:[],scopes:0};
    let lose,finishDecode,decodeEntered;
    const entered=new Promise(resolve=>{decodeEntered=resolve;});
    function decoded(header) {
      const block=format===1023?1:4,blockBytes=[33776,33777,36196,37492].includes(format)?8:format===1023?4:16;
      const mipmaps=[];let w=header.width,h=header.height;
      for(let i=0;i<header.levelCount;i++) {
        mipmaps.push({width:w,height:h,data:new Uint8Array(Math.ceil(w/block)*Math.ceil(h/block)*blockBytes).fill(40+i)});
        w=Math.max(1,Math.floor(w/2));h=Math.max(1,Math.floor(h/2));
      }
      return {image:{width:header.width,height:header.height},flipY:false,premultiplyAlpha:false,type:1009,format,
        colorSpace:header.colorSpace==='srgb'?'srgb':'',mipmaps,
        dispose(){log.temporaryDisposals++;for(const mip of mipmaps)mip.data.fill(255);}};
    }
    const loader={dispose:unused,parse(buffer,onLoad){
      log.parses++;assert.equal(log.scopes,0);
      const header=inspectGltfKtx2(new Uint8Array(buffer));
      // A worker transfers its input. The cached asset must remain intact.
      structuredClone(buffer,{transfer:[buffer]});
      finishDecode=()=>onLoad(decoded(header));decodeEntered();
      if(!late)finishDecode();
    }};
    const device={features:new Set(features),limits:{maxTextureDimension2D:4096},lost:new Promise(resolve=>{lose=resolve;}),destroy:unused,
      pushErrorScope(){log.scopes++;},popErrorScope(){log.scopes--;return Promise.resolve(null);},
      createTexture(descriptor){const texture={descriptor,destroyed:0,createView:view=>({texture,view}),destroy(){this.destroyed++;}};log.textures.push(texture);return texture;},
      createSampler(descriptor){const sampler={descriptor};log.samplers.push(sampler);return sampler;},
      queue:{
        writeTexture(target,data,layout,size){if(log.writes.length===writeErrorAt)throw new Error('upload failed');log.writes.push({target,data:data.slice(),layout,size});},
        copyExternalImageToTexture(...args){log.copies.push(args);},
        async onSubmittedWorkDone(){if(queueError)throw queueError;},
      },
    };
    const gpu={async createGpuDecodedAnimationScene(_device,decoded,options){
      assert.equal(_device,device);assert.equal(log.scopes,0);if(sceneError)throw sceneError;
      for(const drawable of decoded.drawables)assert.ok(drawable.baseColorTexture.view);
      const result={pose:{version:1},view:{cameras:[],lights:[]},cameras:[],lights:[],controller:{},draws:decoded.drawables,deformers:[],
        source:decoded.source,diagnostics:decoded.diagnostics,poseVersion:1,bufferBytes:64,failed:false,disposed:false,exportingEnabled:options.exporting!==false,
        disposals:0,dispose(){if(!this.disposed){this.disposed=true;this.disposals++;}},
        render(){assert.equal(this.disposed,false);log.renders=(log.renders??0)+1;},
        async exportPoseGLB(settings){return {encoded:await settings.resolveTexture(decoded.drawables[0].baseColorTexture),view:settings.sceneView};},
        async whenIdle(){},
      };log.models.push(result);return result;
    }};
    const owner=await link('gltf_scene_loader',{
      './gltf_asset.mjs':{...asset},'./animation_model.mjs':{...model},
      './gltf_textures.mjs':{createGltfTextureResources,GltfTextureError},'./animation_model_gpu.mjs':gpu,
    });
    const images={'color.ktx2':ktx2(),'fallback.png':png};
    const fetch=async(url,options)=>{
      log.fetch.push(url);assert.equal(options.credentials,'omit');assert.equal(options.redirect,'error');assert.equal(log.scopes,0);
      const bytes=images[url.slice(origin.length)];return new Response(bytes??null,{status:bytes===undefined?404:200});
    };
    const bitmap=async()=>({width:1,height:1,close(){log.bitmapCloses++;}});
    const load=(json=fixture(true),options={})=>owner.loadGpuGltfAnimationScene(device,encode(json),{
      ...options,assets:{baseURL:origin,fetch,...options.assets},textures:{ktx2Loader:loader,createImageBitmap:unused,...options.textures},
    });
    return {log,loader,device,load,owner,images,fetch,bitmap,entered,finish:()=>finishDecode(),lose:()=>lose({message:'test device loss'})};
  }

  test('required BasisU reaches owned uploads with exact mips and cached export source',async()=>{
    const h=await harness(),scene=await h.load(fixture(true),{exporting:true});
    assert.deepEqual(h.log.fetch,[origin+'color.ktx2']);assert.equal(h.log.parses,1);assert.equal(h.log.temporaryDisposals,1);
    assert.equal(scene.textureBytes,48);assert.equal(h.log.textures.length,1);
    assert.equal(h.log.textures[0].descriptor.format,'bc7-rgba-unorm-srgb');assert.equal(h.log.textures[0].descriptor.mipLevelCount,3);
    assert.equal(h.log.writes.length,3);
    for(const [i,w]of h.log.writes.entries()) {
      assert.equal(w.target.mipLevel,i);assert.equal(w.layout.bytesPerRow,16);assert.deepEqual(w.size,[4,4,1]);assert.deepEqual([...w.data],Array(16).fill(40+i));
    }
    const exported=await scene.exportPoseGLB();assert.deepEqual(exported.encoded.bytes,ktx2());assert.equal(exported.encoded.mimeType,'image/ktx2');
    assert.equal(exported.encoded.sampler.minFilter,9987);assert.equal(exported.view,scene.view);assert.equal(h.log.fetch.length,1);
    scene.render({});assert.equal(h.log.renders,1);scene.dispose();scene.dispose();
    assert.equal(scene.textureBytes,0);assert.equal(h.log.textures[0].destroyed,1);assert.equal(h.log.models[0].disposals,1);
  });
  test('optional BasisU uses only its core fallback without a decoder',async()=>{
    const h=await harness(),scene=await h.load(fixture(),{textures:{ktx2Loader:null,createImageBitmap:h.bitmap}});
    assert.deepEqual(h.log.fetch,[origin+'fallback.png']);assert.equal(h.log.parses,0);assert.equal(h.log.copies.length,1);assert.equal(h.log.bitmapCloses,1);
    scene.dispose();assert.equal(h.log.textures[0].destroyed,1);
  });
  test('explicit false selects an optional fallback even when a decoder is available',async()=>{
    const h=await harness(),scene=await h.load(fixture(),{decode:{basisu:false},textures:{createImageBitmap:h.bitmap}});
    assert.deepEqual(h.log.fetch,[origin+'fallback.png']);assert.equal(h.log.parses,0);scene.dispose();
  });
  test('required BasisU cannot silently take a fallback without decoder support',async()=>{
    const h=await harness();await assert.rejects(h.load(fixture(true),{textures:{ktx2Loader:null}}),{code:'GLTF_MODEL_UNSUPPORTED'});
    assert.equal(h.log.fetch.length,0);assert.equal(h.log.textures.length,0);
    await assert.rejects(h.load(fixture(true),{decode:{basisu:false}}),{code:'GLTF_MODEL_UNSUPPORTED'});
  });
  test('invalid decoder and capability options fail before model network I/O',async()=>{
    const h=await harness();
    for(const options of [{textures:{ktx2Loader:{}}},{textures:{ktx2Loader:null},decode:{basisu:true}},{decode:{basisu:null}},{decode:{basisu:'true'}}]) {
      await assert.rejects(h.owner.loadGpuGltfAnimationScene(h.device,origin+'model.gltf',{
        assets:{fetch:unused},...options,
      }),{code:'GLTF_MODEL_LOAD_OPTIONS'});
    }
  });
  test('route and decoder settings are captured before an asynchronous fetch',async()=>{
    const h=await harness(),decode={basisu:true},textures={ktx2Loader:h.loader,createImageBitmap:unused};
    const fetch=async(url,settings)=>{
      if(url===origin+'model.gltf') {decode.basisu=false;textures.ktx2Loader=null;return new Response(encode(fixture()));}
      return h.fetch(url,settings);
    };
    const scene=await h.owner.loadGpuGltfAnimationScene(h.device,origin+'model.gltf',{assets:{fetch},decode,textures});
    assert.equal(h.log.parses,1);assert.deepEqual(h.log.fetch,[origin+'color.ktx2']);scene.dispose();
  });
  test('a selected KTX2 URI serving PNG bytes cannot masquerade as a valid source',async()=>{
    const h=await harness();h.images['color.ktx2']=png;
    await assert.rejects(h.load(fixture()),{code:'GLTF_TEXTURE_IMAGE'});
    assert.deepEqual(h.log.fetch,[origin+'color.ktx2']);assert.equal(h.log.parses,0);assert.equal(h.log.textures.length,0);
  });
  test('KTX2 signature alone is not credited as a valid transcodable image',async()=>{
    const h=await harness();h.images['color.ktx2']=ktx2().subarray(0,12);
    await assert.rejects(h.load(),{code:'GLTF_KTX2_HEADER'});assert.equal(h.log.parses,0);assert.equal(h.log.textures.length,0);
  });
  test('asset MIME declarations are checked against KTX2 bytes',async()=>{
    const json={asset:{version:'2.0'},images:[{uri:uri(ktx2(),'image/png')}]};
    const loaded=await asset.loadGltfAsset(encode(json));await assert.rejects(loaded.readImage(0),{code:'GLTF_ASSET_IMAGE'});
  });
  test('one KTX2 image shares storage while texture samplers remain distinct',async()=>{
    const h=await harness(),json=fixture();json.samplers=[{minFilter:9728,wrapS:33071},{minFilter:9987}];json.textures[0].sampler=0;
    json.textures.push({sampler:1,extensions:{[extension]:{source:1}}});
    json.materials.push({extensions:{KHR_materials_unlit:{}},pbrMetallicRoughness:{baseColorTexture:{index:1}}});
    const scene=await h.load(json),[a,b]=scene.draws;
    assert.equal(h.log.textures.length,1);assert.equal(h.log.parses,1);assert.equal(a.baseColorTexture.view,b.baseColorTexture.view);
    assert.notEqual(a.baseColorTexture.sampler,b.baseColorTexture.sampler);
    assert.equal(a.baseColorTexture.sampler.descriptor.lodMaxClamp,0);assert.equal(b.baseColorTexture.sampler.descriptor.lodMaxClamp,undefined);
    assert.equal(scene.textureBytes,48);scene.dispose();
  });
  test('conflicting sRGB/data-map use fails before a foreign decoder runs',async()=>{
    const h=await harness(),json=fixture();delete json.materials[0].extensions;json.materials[0].normalTexture={index:0};
    await assert.rejects(h.load(json),{code:'GLTF_TEXTURE_COLOR'});assert.equal(h.log.parses,0);assert.equal(h.log.textures.length,0);
  });
  test('distinct sRGB and linear images reach the right GPU formats',async()=>{
    const h=await harness(),json=fixture();delete json.materials[0].extensions;
    json.images.push({uri:'normal.ktx2'});json.textures.push({extensions:{[extension]:{source:2}}});json.materials[0].normalTexture={index:1};
    h.images['normal.ktx2']=ktx2({linear:true});const scene=await h.load(json);
    assert.deepEqual(h.log.textures.map(t=>t.descriptor.format),['bc7-rgba-unorm-srgb','bc7-rgba-unorm']);scene.dispose();
  });
  for(const [format,feature,gpuFormat,bytes]of [[33777,'texture-compression-bc','bc1-rgba-unorm-srgb',24],
    [37492,'texture-compression-etc2','etc2-rgb8unorm-srgb',24],[37808,'texture-compression-astc','astc-4x4-unorm-srgb',48],[1023,null,'rgba8unorm-srgb',84]])
    test('retained output '+gpuFormat+' is realized through the owning loader',async()=>{
      const h=await harness({format,features:feature?[feature]:[]}),scene=await h.load();
      assert.equal(h.log.textures[0].descriptor.format,gpuFormat);assert.equal(scene.textureBytes,bytes);scene.dispose();
    });
  test('a device without the selected compressed feature gets no invalid texture',async()=>{
    const h=await harness({features:[]});await assert.rejects(h.load(),{code:'GLTF_KTX2_FEATURE'});
    assert.equal(h.log.textures.length,0);assert.equal(h.log.temporaryDisposals,1);
  });
  test('GPU storage and worst-case transcode expansion retain separate limits',async()=>{
    const h=await harness();await assert.rejects(h.load(fixture(true),{textures:{maxTextureBytes:47}}),{code:'GLTF_TEXTURE_LIMIT'});
    assert.equal(h.log.parses,1);assert.equal(h.log.textures.length,0);
    const next=await harness();await assert.rejects(next.load(fixture(true),{textures:{maxTranscodeBytes:83}}),{code:'GLTF_KTX2_LIMIT'});
    assert.equal(next.log.parses,0);assert.equal(next.log.textures.length,0);
  });
  test('a later upload failure frees both earlier and current owned allocations',async()=>{
    const h=await harness({writeErrorAt:3}),json=fixture();delete json.materials[0].extensions;
    json.images.push({uri:'other.ktx2'});json.textures.push({extensions:{[extension]:{source:2}}});json.materials[0].emissiveTexture={index:1};h.images['other.ktx2']=ktx2();
    await assert.rejects(h.load(json),/upload failed/);assert.equal(h.log.textures.length,2);
    assert.ok(h.log.textures.every(t=>t.destroyed===1));assert.equal(h.log.models.length,0);assert.equal(h.log.scopes,0);
  });
  test('scene-construction failure frees textures and preserves the original error',async()=>{
    const error=new Error('scene failure'),h=await harness({sceneError:error});await assert.rejects(h.load(),e=>e===error);
    assert.equal(h.log.textures[0].destroyed,1);assert.equal(h.log.models.length,0);
  });
  test('queue rejection prevents publication and frees uploaded textures',async()=>{
    const error=new Error('queue failed'),h=await harness({queueError:error});await assert.rejects(h.load(),e=>e===error);
    assert.equal(h.log.textures[0].destroyed,1);assert.equal(h.log.models.length,0);
  });
  test('abort during a retained decode rejects promptly and disposes its late output',async()=>{
    const h=await harness({late:true}),c=new AbortController(),reason=new Error('cancel load');
    const pending=h.load(fixture(true),{signal:c.signal});await h.entered;c.abort(reason);
    await assert.rejects(pending,e=>e===reason);h.finish();assert.equal(h.log.temporaryDisposals,1);
    assert.equal(h.log.textures.length,0);assert.equal(h.log.models.length,0);
  });
  test('device loss during retained decode rejects and cleans late callback output',async()=>{
    const h=await harness({late:true}),pending=h.load();await h.entered;h.lose();
    await assert.rejects(pending,{code:'GLTF_TEXTURE_DEVICE_LOST'});h.finish();assert.equal(h.log.temporaryDisposals,1);assert.equal(h.log.textures.length,0);
  });
  test('device loss after publication makes the scene fail and releases its model',async()=>{
    const h=await harness(),scene=await h.load();h.lose();await Promise.resolve();
    assert.equal(scene.failed,true);assert.throws(()=>scene.render({}),{code:'GLTF_TEXTURE_DEVICE_LOST'});
    assert.equal(h.log.textures[0].destroyed,1);assert.equal(h.log.models[0].disposals,1);scene.dispose();assert.equal(h.log.models[0].disposals,1);
  });
  test('data-URI KTX2 is lazy, cached, and counted once in encoded byte budgets',async()=>{
    const image=ktx2(),json={asset:{version:'2.0'},images:[{uri:uri(image,'image/ktx2')},{uri:'https://unreachable.test/unused.ktx2'}]},input=encode(json);
    const loaded=await asset.loadGltfAsset(input,{fetch:unused});assert.equal(loaded.bytesLoaded,input.length);
    const [a,b]=await Promise.all([loaded.readImage(0),loaded.readImage(0)]);assert.equal(a,b);assert.deepEqual(a.bytes,image);
    assert.equal(a.mimeType,'image/ktx2');assert.equal(loaded.bytesLoaded,input.length+image.length);
    const bounded=await asset.loadGltfAsset(input,{maxBytes:input.length+image.length-1});await assert.rejects(bounded.readImage(0),{code:'GLTF_ASSET_LIMIT'});
  });
  test('embedded GLB KTX2 reaches the decoder at its declared offset with no image fetch',async()=>{
    const h=await harness(),json=fixture(true),image=ktx2(),bin=new Uint8Array(image.length+16);bin.set(image,16);
    json.buffers=[{byteLength:bin.length}];json.bufferViews=[{buffer:0,byteOffset:16,byteLength:image.length}];json.images[1]={bufferView:0,mimeType:'image/ktx2'};
    const scene=await h.owner.loadGpuGltfAnimationScene(h.device,glb(json,bin),{assets:{fetch:unused},textures:{ktx2Loader:h.loader,createImageBitmap:unused},exporting:true});
    assert.equal(h.log.parses,1);const saved=await scene.exportPoseGLB();assert.deepEqual(saved.encoded.bytes,image);scene.dispose();
  });
  test('KTX2 image origins and streamed resource lengths keep existing protections',async()=>{
    const json={asset:{version:'2.0'},images:[{uri:'https://other.test/x.ktx2'}]};
    const blocked=await asset.loadGltfAsset(encode(json),{baseURL:origin,fetch:unused});await assert.rejects(blocked.readImage(0),{code:'GLTF_ASSET_URI'});
    json.images[0].uri='huge.ktx2';let canceled=0;
    const bounded=await asset.loadGltfAsset(encode(json),{baseURL:origin,maxResourceBytes:1024,
      fetch:async()=>new Response(new ReadableStream({start(c){c.enqueue(new Uint8Array(1025));},cancel(){canceled++;}})),
    });await assert.rejects(bounded.readImage(0),{code:'GLTF_ASSET_LIMIT'});assert.equal(canceled,1);
  });
  test('aborting a streaming KTX2 response cancels the pending reader',async()=>{
    const c=new AbortController();let canceled=0,begin;const entered=new Promise(resolve=>{begin=resolve;});
    const loaded=await asset.loadGltfAsset(encode({asset:{version:'2.0'},images:[{uri:'pending.ktx2'}]}),{
      baseURL:origin,signal:c.signal,fetch:async()=>{begin();return new Response(new ReadableStream({cancel(){canceled++;}}));},
    });const pending=loaded.readImage(0);await entered;await Promise.resolve();c.abort();
    await assert.rejects(pending,{name:'AbortError'});assert.equal(canceled,1);
  });
  test('conflicting MIME requirements are rejected before duplicate request elimination',async()=>{
    const h=await harness(),r={textureIndex:0,imageIndex:0,colorSpace:'srgb',image:{mimeType:'image/ktx2'}};
    await assert.rejects(createGltfTextureResources(h.device,[r,{...r,image:{mimeType:'image/png'}}],unused),{code:'GLTF_TEXTURE_REQUEST'});
    assert.equal(h.log.textures.length,0);
  });
  test('explicit MIME requirements cannot be removed by mutating a pending request',async()=>{
    const h=await harness(),r={textureIndex:0,imageIndex:0,colorSpace:'srgb',image:{mimeType:'image/ktx2'}};
    let finish;const pending=createGltfTextureResources(h.device,[r],()=>new Promise(resolve=>{finish=resolve;}),{createImageBitmap:unused});
    r.image.mimeType='image/png';finish({bytes:png,mimeType:'image/png'});
    await assert.rejects(pending,{code:'GLTF_TEXTURE_IMAGE'});assert.equal(h.log.textures.length,0);
  });
  test('an unspecified duplicate use does not erase an explicit MIME requirement',async()=>{
    const h=await harness(),r={textureIndex:0,imageIndex:0,colorSpace:'srgb'};
    await assert.rejects(createGltfTextureResources(h.device,[r,{...r,image:{mimeType:'image/ktx2'}}],async()=>({bytes:png,mimeType:'image/png'})),{code:'GLTF_TEXTURE_IMAGE'});
  });
}
