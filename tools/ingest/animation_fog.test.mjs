import assert from 'node:assert/strict';
import test from 'node:test';
import {FOG_UNIFORM_BYTES, packAnimationFog, animationFogWgsl} from './animation_fog.mjs';
const code = suffix => ({code: 'ANIMATION_FOG_' + suffix});
const linear = (extra = {}) => ({type:'linear', color:[.25,.5,2], near:2, far:10, depthFromClip:[0,0,0,1], ...extra});
const exp2 = (extra = {}) => ({type:'exp2', color:[1,2,3], density:.25, depthFromClip:[0,0,0,1], ...extra});

test('disabled frames fully overwrite every field with zero', () => {
  assert.equal(FOG_UNIFORM_BYTES, 48);
  assert.deepEqual(Array.from(packAnimationFog()), Array(12).fill(0));
  assert.deepEqual(Array.from(packAnimationFog(null)), Array(12).fill(0));
});
test('linear packet has exact native layout, HDR color and independent copied storage', () => {
  const source=linear(), a=packAnimationFog(source), b=packAnimationFog(source);
  assert.equal(a.byteLength,48);assert.deepEqual(Array.from(a),[0,0,0,1,.25,.5,2,0,2,10,0,1]);
  source.color[0]=99;source.depthFromClip[3]=22;a[0]=8;assert.equal(a[4],.25);assert.equal(a[3],1);assert.equal(b[0],0);
});
test('exp2 density and selector use separate words; unused near/far are zero', () => {
  assert.deepEqual(Array.from(packAnimationFog(exp2())),[0,0,0,1,1,2,3,0,0,0,.25,2]);
});
test('ordinary numeric typed arrays are copied with finite f32 rounding', () => {
  const p=packAnimationFog(linear({color:new Float64Array([.1,.2,.3]),depthFromClip:new Float32Array([1,2,3,4])}));
  assert.deepEqual(Array.from(p.slice(0,4)),[1,2,3,4]);assert.equal(p[4],Math.fround(.1));
});
for(const [name,value,suffix] of [
  ['false',false,'PROFILE'],['array',[],'PROFILE'],['unknown profile',{type:'radial'},'PROFILE'],
  ['extraneous density',linear({density:1}),'PROFILE'],['extraneous near',exp2({near:1}),'PROFILE'],
  ['negative color',linear({color:[-1,0,0]}),'VALUE'],['infinite color',linear({color:[Infinity,0,0]}),'VALUE'],
  ['color overflow',linear({color:[1e40,0,0]}),'VALUE'],['missing color',linear({color:undefined}),'VALUE'],
  ['wrong color length',linear({color:[1,2]}),'VALUE'],['nonnumeric color',linear({color:['1',0,0]}),'VALUE'],
  ['bigint color',linear({color:new BigInt64Array(3)}),'VALUE'],
  ['missing depth row',linear({depthFromClip:undefined}),'VALUE'],['depth overflow',linear({depthFromClip:[0,0,1e40,0]}),'VALUE'],
  ['nan depth',linear({depthFromClip:[0,0,NaN,0]}),'VALUE'],['dataview',linear({depthFromClip:new DataView(new ArrayBuffer(16))}),'VALUE'],
  ['equal edges',linear({near:2,far:2}),'VALUE'],['reversed edges',linear({near:4,far:2}),'VALUE'],
  ['edges collapse after f32 rounding',linear({near:1,far:1+2**-25}),'VALUE'],
  ['edge difference overflow',linear({near:-3e38,far:3e38}),'VALUE'],
  ['negative density',exp2({density:-.1}),'VALUE'],['infinite density',exp2({density:Infinity}),'VALUE'],
  ['nan density',exp2({density:NaN}),'VALUE'],['missing density',exp2({density:undefined}),'VALUE'],
])test('reject '+name,()=>assert.throws(()=>packAnimationFog(value),code(suffix)));
test('zero density, zero/negative linear near and representable tiny intervals are admitted',()=>{
  assert.equal(packAnimationFog(exp2({density:0}))[10],0);
  assert.equal(packAnimationFog(linear({near:-2,far:0}))[8],-2);
  assert.equal(packAnimationFog(linear({near:0,far:2**-100}))[9],2**-100);
});
test('shader computes distance fog on shaded RGB without alpha, texture, depth-buffer or output-transfer effects',()=>{
  const s=animationFogWgsl();assert.match(s,/@group\(0\) @binding\(1\)/);assert.match(s,/depth_from_clip: vec4<f32>/);
  assert.match(s,/smoothstep\(/);assert.match(s,/exp\(-optical_depth \* optical_depth\)/);
  assert.match(s,/return mix\(color, fog_info.color.rgb, factor\)/);
  assert.doesNotMatch(s,/texture|frag_depth|discard|color\.a|pow\(|tonemap|srgb/i);
  assert.match(animationFogWgsl(7),/@binding\(7\)/);
  for(const x of [-1,.5,65536,NaN,'1'])assert.throws(()=>animationFogWgsl(x),code('PROFILE'));
});
test('saturation threshold is beyond binary32 rounding to full fog',()=>{
  assert.equal(Math.fround(1-Math.exp(-(4.25**2))),1);
  assert.notEqual(Math.fround(1-Math.exp(-(4**2))),1);
});
