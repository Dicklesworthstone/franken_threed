/** Factory boundary tests, not native canvas or retained-Three execution. */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
const key='__f3d_canvas_deformation_test__';
const module=code=>'data:text/javascript,'+encodeURIComponent(code);
const direct=module(`export const createGpuCanvasRenderer=(...a)=>globalThis.${key}.canvas('direct',...a);`);
const hdr=module(`export const createGpuHdrCanvasRenderer=(...a)=>globalThis.${key}.canvas('hdr',...a);`);
const scene=module(`export const createGpuThreeScene=(...a)=>globalThis.${key}.scene(...a);`);
const errors=module(`export class GpuCanvasError extends Error {constructor(code,message){super(message);this.code='GPU_CANVAS_'+code;}}`);
const text=(await fs.readFile(new URL('./three_canvas.mjs',import.meta.url),'utf8'))
  .replace("'./gpu_canvas_renderer.mjs'",JSON.stringify(direct))
  .replace("'./gpu_hdr_canvas.mjs'",JSON.stringify(hdr))
  .replace("'./three_scene.mjs'",JSON.stringify(scene))
  .replace("'./gpu_canvas.mjs'",JSON.stringify(errors));
const {createGpuThreeCanvas,createGpuThreeHdrCanvas}=await import(module(text));
for(const [kind,factory] of [['direct',createGpuThreeCanvas],['hdr',createGpuThreeHdrCanvas]]){
  test(`${kind} canvas snapshots deformation options and forwards its owned lifetime`,async()=>{
    let continueCreation,received;
    const device={},canvas={},source={},three={REVISION:'186'},abort=new AbortController(),lifetime=new AbortController();
    const attachments={format:kind==='hdr'?'rgba16float':'bgra8unorm-srgb',depthFormat:'depth24plus',sampleCount:1};
    globalThis[key]={
      canvas(k,c,make,options){assert.equal(k,kind);assert.equal(c,canvas);assert.equal(options.signal,abort.signal);
        return new Promise(resolve=>{continueCreation=()=>resolve(make(device,attachments,{signal:lifetime.signal}));});},
      scene(d,s,options){assert.equal(d,device);assert.equal(s,source);received=options;return {ready:true};},
    };
    const options={three,signal:abort.signal,scene:{deformation:{maxJoints:24},maxDeformedMeshes:8}};
    const pending=factory(canvas,source,options);options.scene.deformation.maxJoints=200;options.scene.maxDeformedMeshes=50;
    continueCreation();assert.deepEqual(await pending,{ready:true});
    assert.equal(received.deformation.maxJoints,24);assert.equal(received.maxDeformedMeshes,8);
    assert.equal(received.signal,lifetime.signal);assert.equal(received.three,three);assert.deepEqual(received.renderer,attachments);
  });
  test(`${kind} canvas refuses competing lifetime owners and malformed deformation options`,()=>{
    globalThis[key]={canvas(){assert.fail('must reject before canvas creation');}};
    for(const source of [{signal:new AbortController().signal},{deformation:null},{deformation:[]}])
      assert.throws(()=>factory({}, {}, {scene:source}),{code:'GPU_CANVAS_OPTIONS'});
  });
}
