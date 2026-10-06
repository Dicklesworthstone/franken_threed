/** The vendored r186 DFG table decodes to the same values as the pinned
 * upstream DataTexture source, and the WGSL sampler is emitted once. */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import {ANIMATION_DFG_LUT, animationDfgWgsl} from './animation_dfg.mjs';

test('vendored DFG halves match the pinned upstream table bit for bit', () => {
  const source = fs.readFileSync(path.resolve(process.env.F3D_THREE_ROOT ?? 'upstream/three.js', 'src/nodes/functions/BSDF/DFGLUT.js'), 'utf8');
  const body = source.slice(source.indexOf('new Uint16Array( ['), source.indexOf('] );'));
  const halves = [...body.matchAll(/0x[0-9a-f]+/gi)].map(m => Number(m[0]));
  assert.equal(halves.length, 512);
  const view = new DataView(new ArrayBuffer(2));
  const decode = h => { // independent half decode through Float32 bit assembly
    const s = h >> 15, e = (h >> 10) & 31, f = h & 1023;
    if (e === 0) return (s ? -1 : 1) * f * 2 ** -24;
    return (s ? -1 : 1) * (1 + f / 1024) * 2 ** (e - 15);
  };
  assert.deepEqual(ANIMATION_DFG_LUT, halves.map(decode));
  assert.ok(ANIMATION_DFG_LUT.every(v => v >= 0 && v <= 1));
  void view;
});

test('WGSL table is a module-scope constant with a clamp-to-edge bilinear sampler', () => {
  const wgsl = animationDfgWgsl();
  assert.match(wgsl, /^\nconst f3d_dfg_lut = array<vec2<f32>, 256>\(/);
  assert.match(wgsl, /fn f3d_dfg\(roughness: f32, dot_nv: f32\) -> vec2<f32>/);
  assert.ok(!wgsl.includes('var<private>'), 'no per-invocation table initialization');
});
