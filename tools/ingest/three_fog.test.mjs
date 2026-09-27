/** Production source capture/routing with explicit source and native fixtures.
 * Matrix inversion here is an independent Gauss-Jordan test implementation;
 * these are numerical/command tests, not retained-Three or GPU pixel tests. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {inspectThreeFog,threeFogDescriptor,withThreeFogReceivers} from './three_fog.mjs';
class Color {constructor(r=1,g=1,b=1){Object.assign(this,{r,g,b,isColor:true});}}
class Fog {constructor(near=1,far=100){Object.assign(this,{isFog:true,near,far,color:new Color()});}}
class FogExp2 {constructor(density=.02){Object.assign(this,{isFogExp2:true,density,color:new Color()});}}
class Matrix4 {
  constructor(e=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]){this.elements=e;}
  copy(m){this.elements=Array.from(m.elements);return this;}
  solve(inverse){
    const rows=Array.from({length:4},(_,r)=>Array.from({length:8},(_,c)=>c<4?this.elements[c*4+r]:+(c-4===r)));let det=1;
    for(let i=0;i<4;i++){
      let p=i;for(let r=i+1;r<4;r++)if(Math.abs(rows[r][i])>Math.abs(rows[p][i]))p=r;
      if(rows[p][i]===0)return inverse?Array(16).fill(0):0;
      if(p!==i){[rows[p],rows[i]]=[rows[i],rows[p]];det=-det;}
      const pivot=rows[i][i];det*=pivot;for(let c=0;c<8;c++)rows[i][c]/=pivot;
      for(let r=0;r<4;r++)if(r!==i){const f=rows[r][i];for(let c=0;c<8;c++)rows[r][c]-=f*rows[i][c];}
    }
    return inverse?Array.from({length:16},(_,i)=>rows[i%4][4+Math.floor(i/4)]):det;
  }
  determinant(){return this.solve(false);}invert(){this.elements=this.solve(true);return this;}
}
class Camera {constructor(){this.isPerspectiveCamera=true;this.coordinateSystem=2000;this.projectionMatrix=projection(false,false);}}
const THREE={REVISION:'186',Color,Fog,FogExp2,Matrix4,Camera,WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001};
function projection(ortho,gpu,{near=.1,far=1000,aspect=1.7,offsetX=.2,offsetY=-.3}={}){
  return new Matrix4(ortho?[.3,0,0,0,0,.7,0,0,0,0,(gpu?1:2)/(near-far),0,offsetX,offsetY,(gpu?near:far+near)/(near-far),1]:
    [1.3/aspect,0,0,0,0,1.3,0,0,offsetX,offsetY,(gpu?far:far+near)/(near-far),-1,0,0,(gpu?far:2*far)*near/(near-far),0]);
}
function clip(p,v,gpu){const e=p.elements,q=Array.from({length:4},(_,r)=>e[r]*v[0]+e[4+r]*v[1]+e[8+r]*v[2]+e[12+r]*v[3]);if(!gpu)q[2]=.5*(q[2]+q[3]);return q;}
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const near=(a,b)=>assert.ok(Math.abs(a-b)<=1e-7*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const scene=()=>({fog:new Fog()});
const code=s=>({code:'THREE_FOG_'+s});

test('source linear/exp2 capture copies live linear color and admits null',()=>{
  const s=scene();s.fog.color=new Color(.1,.5,4);assert.deepEqual(inspectThreeFog(s.fog,THREE),{type:'linear',color:[.1,.5,4],near:1,far:100});
  s.fog=new FogExp2(.3);assert.deepEqual(inspectThreeFog(s.fog,THREE),{type:'exp2',color:[1,1,1],density:.3});assert.equal(inspectThreeFog(null,THREE),null);
  assert.equal(threeFogDescriptor({fog:null},null,THREE),null);
});
for(const orthographic of [false,true])for(const gpu of [false,true])test(`native depth row recovers view Z for ${orthographic?'orthographic':'perspective'} ${gpu?'WebGPU':'WebGL'} projection`,()=>{
  const s=scene(),c=new Camera();c.isPerspectiveCamera=!orthographic;c.isOrthographicCamera=orthographic;c.coordinateSystem=gpu?2001:2000;
  for(const [nearZ,far] of [[.1,1000],[1,2],[100,10000]]){
    c.projectionMatrix=projection(orthographic,gpu,{near:nearZ,far});const before=Array.from(c.projectionMatrix.elements),f=threeFogDescriptor(s,c,THREE);
    for(const point of [[0,0,-nearZ,1],[10,-7,-far,1],[-4,2,-.5*(nearZ+far),1]])near(dot(f.depthFromClip,clip(c.projectionMatrix,point,gpu)),-point[2]);
    assert.deepEqual(c.projectionMatrix.elements,before);assert.ok(Object.isFrozen(f));assert.ok(Object.isFrozen(f.color));assert.ok(Object.isFrozen(f.depthFromClip));
  }
});
test('capture changes with live fog parameters, not camera world translation or unrelated inverse caches',()=>{
  const s=scene(),c=new Camera(),a=threeFogDescriptor(s,c,THREE);c.matrixWorld={elements:Array(16).fill(99)};c.projectionMatrixInverse={elements:Array(16).fill(0)};
  assert.deepEqual(threeFogDescriptor(s,c,THREE),a);s.fog.near=3;s.fog.far=20;s.fog.color.r=.4;const b=threeFogDescriptor(s,c,THREE);
  assert.equal(b.near,3);assert.equal(a.near,1);assert.equal(a.color[0],1);assert.equal(b.color[0],.4);s.fog=new FogExp2(.5);assert.equal(threeFogDescriptor(s,c,THREE).type,'exp2');
});
test('un-divided vertex depth has correct perspective interpolation rather than screen-Z interpolation',()=>{
  const c=new Camera(),f=threeFogDescriptor(scene(),c,THREE),depths=[2,20,50],weights=[.2,.3,.5];
  const clips=depths.map((z,i)=>clip(c.projectionMatrix,[i-1,1-i,-z,1],false));
  const invW=weights.reduce((v,w,i)=>v+w/clips[i][3],0);
  const interpolated=weights.reduce((v,w,i)=>v+w*dot(f.depthFromClip,clips[i])/clips[i][3],0)/invW;
  near(interpolated,1/invW);assert.notEqual(interpolated,dot(weights,depths));
});
for(const [name,edit,suffix] of [
  ['foreign source',s=>{s.fog={isFog:true,color:new Color(),near:1,far:2};},'SOURCE'],
  ['bad color',s=>{s.fog.color={r:1,g:1,b:1};},'SOURCE'],
  ['conflicting flags',s=>{s.fog.isFogExp2=true;},'SOURCE'],
  ['array camera',(s,c)=>{c.isArrayCamera=true;},'CAMERA'],
  ['reversed camera',(s,c)=>{c.reversedDepth=true;},'CAMERA'],
  ['foreign camera',(s,c)=>{Object.setPrototypeOf(c,{});},'CAMERA'],
  ['bad convention',(s,c)=>{c.coordinateSystem=17;},'CAMERA'],
  ['singular projection',(s,c)=>{c.projectionMatrix.elements.fill(0);},'CAMERA'],
  ['nan projection',(s,c)=>{c.projectionMatrix.elements[0]=NaN;},'CAMERA'],
])test('source refuses '+name,()=>{const s=scene(),c=new Camera();edit(s,c);assert.throws(()=>threeFogDescriptor(s,c,THREE),code(suffix));});
test('source profiles reject invalid density and f32-collapsed linear intervals',()=>{
  assert.throws(()=>inspectThreeFog(new Fog(1,1+2**-25),THREE),{code:'ANIMATION_FOG_VALUE'});
  assert.throws(()=>inspectThreeFog(new FogExp2(-.5),THREE),{code:'ANIMATION_FOG_VALUE'});
  assert.throws(()=>inspectThreeFog(null,{...THREE,REVISION:'185'}),code('SOURCE'));
});
function renderer(){const frames=[];const native={frames,disposed:false,failed:false,allocatedBytes:64,drawCallCount:0,bundleDiagnostics:{reuses:0},
  addMesh(g,o){return {g,o};},render(f){native.hook?.(f);frames.push(f);native.drawCallCount=f.draws.length;},
  async whenIdle(){return native;},dispose(){native.disposed=true;}};return native;}
const draw=(id,receiveFog=true)=>({id,receiveFog,mesh:{id}});
const frame=(draws,fog=threeFogDescriptor(scene(),new Camera(),THREE))=>({draws,fog,colorView:{},depthView:{},resolveTarget:{},loadOp:'clear',depthLoadOp:'clear',lighting:{lights:[]},shadow:{},environment:{}});
test('routing only splits adjacent receiver spans, preserves draws and forwards all other frame services',()=>{
  const n=renderer(),o=withThreeFogReceivers(n),f=frame([draw(0),draw(1),draw(2,false),draw(3),draw(4)]);o.render(f);
  assert.deepEqual(n.frames.map(f=>f.draws.map(d=>d.id)),[[0,1],[2],[3,4]]);assert.deepEqual(n.frames.map(f=>!!f.fog),[true,false,true]);
  assert.deepEqual(n.frames.map(f=>f.loadOp),['clear','load','load']);assert.deepEqual(n.frames.map(f=>f.depthLoadOp),['clear','load','load']);
  for(const output of n.frames){for(const d of output.draws)assert.equal(d.receiveFog,undefined);for(const key of ['lighting','shadow','environment','resolveTarget','colorView','depthView'])assert.equal(output[key],f[key]);}
  assert.equal(o.drawCount,5);assert.equal(o.drawCallCount,5);assert.equal(o.colorPassCount,3);assert.equal(o.allocatedBytes,64);assert.equal(o.bundleDiagnostics,n.bundleDiagnostics);
});
test('no fog combines all flags into one ordinary pass and empty scenes still clear once',()=>{
  const n=renderer(),o=withThreeFogReceivers(n);o.render(frame([draw(1),draw(2,false),draw(3)],null));assert.equal(n.frames.length,1);assert.equal(n.frames[0].fog,null);
  o.render(frame([]));assert.equal(n.frames.length,2);assert.equal(o.colorPassCount,1);assert.equal(o.drawCount,0);
});
test('all receiver flags and fog fields preflight before any partial submission',()=>{
  const n=renderer(),o=withThreeFogReceivers(n,3);assert.throws(()=>o.render(frame([draw(1),draw(2,'no')])),code('FRAME'));
  assert.throws(()=>o.render(frame([draw(1)],{type:'linear'})),{code:'ANIMATION_FOG_VALUE'});
  assert.throws(()=>o.render(frame([draw(1),draw(2),draw(3),draw(4)])),code('LIMIT'));assert.equal(n.frames.length,0);assert.equal(o.failed,false);
});
test('receiver routing snapshots mutable fog values before the first downstream submission',()=>{
  const n=renderer(),o=withThreeFogReceivers(n),f=frame([draw(1),draw(2,false),draw(3)],{type:'linear',color:[1,0,0],depthFromClip:[0,0,0,1],near:1,far:20});
  n.hook=()=>{f.fog.color[0]=0;f.fog.far=100;};o.render(f);assert.equal(n.frames[2].fog.color[0],1);assert.equal(n.frames[2].fog.far,20);
});
test('a pre-submit failure stays recoverable, a partial submission failure is terminal',()=>{
  const n=renderer(),o=withThreeFogReceivers(n),f=frame([draw(1),draw(2,false)]);n.hook=()=>{throw Error('preflight');};
  assert.throws(()=>o.render(f),/preflight/);assert.equal(o.failed,false);n.hook=()=>{if(n.frames.length)throw Error('second span');};
  assert.throws(()=>o.render(f),/second span/);assert.equal(o.failed,true);assert.equal(n.disposed,true);assert.throws(()=>o.render(f),/second span/);
});
test('reentrant submission/disposal refuses without disrupting the outer frame',async()=>{
  const n=renderer(),o=withThreeFogReceivers(n),f=frame([draw(1)]);n.hook=()=>{assert.throws(()=>o.render(f),code('REENTRANT'));assert.throws(()=>o.dispose(),code('REENTRANT'));};
  o.render(f);await o.whenIdle();assert.equal(o.failed,false);o.dispose();assert.throws(()=>o.render(f),code('DISPOSED'));
});
