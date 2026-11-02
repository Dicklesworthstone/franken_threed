/** Ordered private state contracts; browser MaskPass parity proves pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {registerThreePassState, getThreePassState, getThreeStencilConstants, captureThreePassState,
  beginThreePassState, applyThreePassMaterial, applyThreePassClear, finishThreePassState,
  recordThreePassMaterial, recordThreePassClear} from './three_pass_state.mjs';

const material = overrides => ({colorWrite: true, depthWrite: true, stencilWrite: false,
  stencilWriteMask: 0xff, stencilFunc: T.AlwaysStencilFunc, stencilRef: 0, stencilFuncMask: 0xff,
  stencilFail: T.KeepStencilOp, stencilZFail: T.KeepStencilOp, stencilZPass: T.KeepStencilOp, ...overrides});
const raster = Object.freeze({writeMask: 15, depthWriteEnabled: true, depthCompare: 'less-equal',
  blend: null, cullMode: 'back', frontFace: 'ccw'});
const frame = Object.freeze({loadOp: 'clear', depthLoadOp: 'clear', stencilLoadOp: 'clear',
  clearColor: Object.freeze([0.1, 0.2, 0.3, 1]), clearDepth: 1, clearStencil: 0});
function fixture() {
  const renderer = {};
  const state = registerThreePassState(renderer, T), constants = getThreeStencilConstants(renderer);
  return {renderer, state, constants, capture: () => captureThreePassState(renderer)};
}

test('private registration never exposes or consults a fabricated public context or state', () => {
  const renderer = {getContext() { throw Error('public native context remains unavailable'); }};
  Object.defineProperty(renderer, 'state', {get() { throw Error('public state remains unavailable'); }});
  const state = registerThreePassState(renderer, T);
  assert.equal(getThreePassState(renderer), state); assert.equal(registerThreePassState(renderer, T), state);
  assert.equal(getThreeStencilConstants(renderer).NOTEQUAL, T.NotEqualStencilFunc);
  assert.equal(getThreeStencilConstants(renderer).REPLACE, T.ReplaceStencilOp);
  assert.throws(() => renderer.getContext(), /unavailable/); assert.throws(() => renderer.state, /unavailable/);
  const exactState = {}, exactContext = {}, exact = {state: exactState, getContext: () => exactContext};
  assert.equal(getThreePassState(exact), exactState); assert.equal(getThreeStencilConstants(exact), exactContext);
  assert.equal(captureThreePassState(exact), null);
  assert.equal(beginThreePassState(null), null); assert.equal(finishThreePassState(null), null);
  assert.equal(applyThreePassMaterial(null, material(), raster), raster); assert.equal(applyThreePassClear(null, frame), frame);
});

test('ordinary renderers stay inactive until private transformed state or constants access', () => {
  for (const activate of [getThreePassState, getThreeStencilConstants]) {
    const renderer = {}, state = registerThreePassState(renderer, T);
    assert.equal(captureThreePassState(renderer), null);
    state.buffers.stencil.setClear(11);
    recordThreePassMaterial(renderer, material({stencilWrite: true, stencilWriteMask: 3}));
    recordThreePassClear(renderer, {}, {stencil: true});
    assert.equal(captureThreePassState(renderer), null, 'registration, public clear value and baseline hooks never activate');
    activate(renderer);
    const token = captureThreePassState(renderer), values = beginThreePassState(token);
    assert.equal(values.stencilClear, 11); assert.equal(values.stencilWriteMask, 0xffffffff);
    finishThreePassState(token);
  }
});

test('first private token inherits an earlier ordinary draw at execution, preserving accepted explicit writes', () => {
  const renderer = {}, registered = registerThreePassState(renderer, T);
  assert.equal(captureThreePassState(renderer), null, 'the earlier accepted render has no pass token');
  const state = getThreePassState(renderer);
  assert.equal(state, registered);
  state.buffers.stencil.setLocked(true);
  const first = captureThreePassState(renderer);
  state.buffers.stencil.setFunc(T.NotEqualStencilFunc, 4, 0x2f);
  const second = captureThreePassState(renderer);
  recordThreePassMaterial(renderer, material({stencilWrite: true, stencilFunc: T.EqualStencilFunc,
    stencilRef: 17, stencilFuncMask: 0x19, stencilWriteMask: 0x37, stencilZPass: T.ReplaceStencilOp,
    colorWrite: false, depthWrite: false}));
  const inherited = applyThreePassMaterial(first, material(), raster);
  assert.equal(inherited.stencil.front.compare, 'equal'); assert.equal(inherited.stencilReference, 17);
  assert.equal(inherited.stencil.readMask, 0x19); assert.equal(inherited.stencil.writeMask, 0x37);
  assert.equal(inherited.stencil.front.passOp, 'replace');
  assert.equal(inherited.writeMask, 15); assert.equal(inherited.depthWriteEnabled, true);
  finishThreePassState(first);
  const explicit = applyThreePassMaterial(second, material(), raster);
  assert.equal(explicit.stencil.front.compare, 'not-equal'); assert.equal(explicit.stencilReference, 4);
  assert.equal(explicit.stencil.readMask, 0x2f); assert.equal(explicit.stencil.writeMask, 0x37);
  assert.equal(explicit.stencil.front.passOp, 'replace'); finishThreePassState(second);
});

test('ordinary material disable and attachment-free clear update baseline without losing pending explicit state', () => {
  const renderer = {}, registered = registerThreePassState(renderer, T);
  recordThreePassMaterial(renderer, material({stencilWrite: true, stencilRef: 23, stencilWriteMask: 0x12}));
  const inactive = {stencilWrite: false};
  for (const key of ['stencilFunc', 'stencilRef', 'stencilFuncMask', 'stencilWriteMask', 'stencilFail', 'stencilZFail', 'stencilZPass'])
    Object.defineProperty(inactive, key, {get() { throw Error('inactive field read'); }});
  recordThreePassMaterial(renderer, inactive);
  getThreePassState(renderer);
  registered.buffers.stencil.setMask(0x55); registered.buffers.stencil.setLocked(true);
  const token = captureThreePassState(renderer);
  recordThreePassClear(renderer, {}, {stencil: true});
  const values = beginThreePassState(token);
  assert.equal(values.stencilTest, false); assert.equal(values.stencilRef, 23);
  assert.equal(values.stencilWriteMask, 0x55, 'earlier queued clear cannot overwrite an accepted explicit write');
  assert.equal(applyThreePassMaterial(token, material(), raster), raster); finishThreePassState(token);
  const other = {}, otherState = registerThreePassState(other, T);
  recordThreePassMaterial(other, material({stencilWrite: true, stencilWriteMask: 0x12}));
  getThreePassState(other); otherState.buffers.stencil.setLocked(true);
  const afterClear = captureThreePassState(other);
  recordThreePassClear(other, {}, {stencil: true});
  assert.equal(beginThreePassState(afterClear).stencilWriteMask, 0xffffffff);
  finishThreePassState(afterClear);
});

test('buffer locks suppress mask/test writes while stencil function, operations and clear remain live', () => {
  const {state, capture, constants: gl} = fixture();
  state.buffers.color.setMask(false); state.buffers.depth.setMask(false);
  state.buffers.color.setLocked(true); state.buffers.depth.setLocked(true);
  state.buffers.color.setMask(true); state.buffers.depth.setMask(true);
  state.buffers.stencil.setTest(true); state.buffers.stencil.setMask(0x7f); state.buffers.stencil.setLocked(true);
  state.buffers.stencil.setTest(false); state.buffers.stencil.setMask(0);
  state.buffers.stencil.setFunc(gl.NOTEQUAL, 9, 0x35);
  state.buffers.stencil.setOp(gl.REPLACE, gl.INCR, gl.DECR_WRAP); state.buffers.stencil.setClear(0x123);
  const token = capture(), values = beginThreePassState(token);
  assert.equal(values.colorMask, false); assert.equal(values.depthMask, false);
  assert.equal(values.stencilTest, true); assert.equal(values.stencilWriteMask, 0x7f);
  assert.equal(values.stencilFunc, gl.NOTEQUAL); assert.equal(values.stencilClear, 0x123);
  const result = applyThreePassMaterial(token, material(), raster);
  assert.equal(result.writeMask, 0); assert.equal(result.depthWriteEnabled, false);
  assert.equal(result.stencil.front.compare, 'not-equal'); assert.equal(result.stencilReference, 9);
  assert.deepEqual(result.stencil.front, {compare: 'not-equal', failOp: 'replace', depthFailOp: 'increment-clamp', passOp: 'decrement-wrap'});
  assert.equal(result.stencil.readMask, 0x35); assert.equal(result.stencil.writeMask, 0x7f);
  const final = finishThreePassState(token);
  assert.equal(final.colorMask, false); assert.equal(final.depthMask, false);
  assert.equal(raster.writeMask, 15); assert.equal(raster.stencil, undefined); assert.ok(Object.isFrozen(result));
});

test('a locked stencil-writing material changes func/ref/operations but inherits the locked write mask', () => {
  const {state, capture, constants: gl} = fixture();
  state.buffers.stencil.setTest(true); state.buffers.stencil.setMask(0x31); state.buffers.stencil.setLocked(true);
  const token = capture();
  const result = applyThreePassMaterial(token, material({stencilWrite: true, stencilWriteMask: 0,
    stencilFunc: gl.LESS, stencilRef: 300, stencilFuncMask: 0x65, stencilFail: gl.ZERO,
    stencilZFail: gl.INVERT, stencilZPass: gl.REPLACE}), raster);
  assert.equal(result.stencil.writeMask, 0x31); assert.equal(result.stencil.readMask, 0x65);
  assert.equal(result.stencilReference, 255); assert.equal(result.stencil.front.compare, 'less');
  assert.deepEqual(result.stencil.front, {compare: 'less', failOp: 'zero', depthFailOp: 'invert', passOp: 'replace'});
  const inactive = {colorWrite: true, depthWrite: true, stencilWrite: false};
  for (const key of ['stencilFunc', 'stencilRef', 'stencilFuncMask', 'stencilWriteMask', 'stencilFail', 'stencilZFail', 'stencilZPass'])
    Object.defineProperty(inactive, key, {get() { throw Error('inactive field read'); }});
  assert.equal(applyThreePassMaterial(token, inactive, result), result, 'an inherited equal raster reuses its immutable descriptor');
  finishThreePassState(token);
});

test('queued tokens inherit executed material state before applying later explicit writes', () => {
  const {state, capture, constants: gl} = fixture(), first = capture();
  state.buffers.stencil.setLocked(true);
  const second = capture();
  state.buffers.stencil.setFunc(gl.NOTEQUAL, 1, 0xffffffff);
  const third = capture();
  assert.throws(() => beginThreePassState(second), {code: 'THREE_PASS_STATE_ORDER'});
  const initial = applyThreePassMaterial(first, material({stencilWrite: true, stencilRef: 37,
    stencilFunc: gl.EQUAL, stencilFuncMask: 0x71, stencilWriteMask: 0x12, stencilZPass: gl.REPLACE}), raster);
  finishThreePassState(first);
  const inherited = applyThreePassMaterial(second, material(), raster);
  assert.equal(inherited.stencilReference, 37); assert.equal(inherited.stencil.front.compare, 'equal');
  assert.equal(inherited.stencil.writeMask, 0x12); assert.equal(inherited.stencil.front.passOp, 'replace');
  assert.deepEqual(inherited.stencil, initial.stencil); finishThreePassState(second);
  const copied = applyThreePassMaterial(third, material(), raster);
  assert.equal(copied.stencilReference, 1); assert.equal(copied.stencil.front.compare, 'not-equal');
  assert.equal(copied.stencil.writeMask, 0x12); finishThreePassState(third);
});

test('a locked disabled test strips compiled stencil state but retains material values for later requests', () => {
  const {state, capture, constants: gl} = fixture();
  state.buffers.stencil.setLocked(true);
  const first = capture(), stencilMaterial = material({stencilWrite: true, stencilRef: 23, stencilFunc: gl.EQUAL});
  const compiled = Object.freeze({...raster, stencil: {front: {}, back: {}, readMask: 255, writeMask: 255}, stencilReference: 23});
  const disabled = applyThreePassMaterial(first, stencilMaterial, compiled);
  assert.equal(disabled.stencil, undefined); assert.equal(disabled.stencilReference, undefined);
  finishThreePassState(first);
  state.buffers.stencil.setLocked(false); state.buffers.stencil.setTest(true); state.buffers.stencil.setLocked(true);
  const second = capture(), inherited = applyThreePassMaterial(second, material(), raster);
  assert.equal(inherited.stencilReference, 23); assert.equal(inherited.stencil.front.compare, 'equal');
  finishThreePassState(second);
});

test('same-value explicit writes still override an earlier queued material mutation', () => {
  const {state, capture, constants: gl} = fixture(), first = capture();
  state.buffers.stencil.setTest(false); state.buffers.stencil.setMask(0xffffffff);
  state.buffers.stencil.setFunc(gl.ALWAYS, 0, 0xffffffff); state.buffers.stencil.setOp(gl.KEEP, gl.KEEP, gl.KEEP);
  const second = capture();
  applyThreePassMaterial(first, material({stencilWrite: true, stencilRef: 29, stencilFunc: gl.LESS,
    stencilFuncMask: 1, stencilWriteMask: 2, stencilFail: gl.INVERT, stencilZPass: gl.REPLACE}), raster);
  finishThreePassState(first);
  const values = beginThreePassState(second);
  assert.equal(values.stencilTest, false); assert.equal(values.stencilWriteMask, 0xffffffff);
  assert.equal(values.stencilFunc, gl.ALWAYS); assert.equal(values.stencilRef, 0);
  assert.equal(values.stencilFuncMask, 0xffffffff); assert.equal(values.stencilFail, gl.KEEP); assert.equal(values.stencilZPass, gl.KEEP);
  assert.equal(applyThreePassMaterial(second, material(), raster), raster); finishThreePassState(second);
});

test('render completion resets only unlocked color/depth masks and clear-only completion preserves them', () => {
  const {state, capture} = fixture(), first = capture();
  applyThreePassMaterial(first, material({colorWrite: false, depthWrite: false, stencilWrite: true, stencilWriteMask: 0x13}), raster);
  const final = finishThreePassState(first);
  assert.equal(final.colorMask, true); assert.equal(final.depthMask, true); assert.equal(final.stencilWriteMask, 0x13);
  assert.equal(finishThreePassState(first), final);
  assert.throws(() => beginThreePassState(first), {code: 'THREE_PASS_STATE_ORDER'});
  state.buffers.color.setMask(false); state.buffers.depth.setMask(false);
  const second = capture();
  const clearFinal = finishThreePassState(second, {render: false});
  assert.equal(clearFinal.colorMask, false); assert.equal(clearFinal.depthMask, false);
  const third = capture();
  assert.equal(beginThreePassState(third).colorMask, false); finishThreePassState(third);
});

test('cancelled queued requests retain explicit assignments without blocking later requests', () => {
  const {state, capture} = fixture(), first = capture();
  state.buffers.stencil.setFunc(T.NotEqualStencilFunc, 7, 0x21);
  state.buffers.color.setMask(false); state.buffers.depth.setMask(false);
  const cancelled = capture();
  state.buffers.stencil.setLocked(true);
  const next = capture();
  assert.throws(() => finishThreePassState(cancelled), {code: 'THREE_PASS_STATE_ORDER'});
  assert.equal(finishThreePassState(cancelled, {render: false}), null);
  assert.throws(() => beginThreePassState(cancelled), {code: 'THREE_PASS_STATE_ORDER'});
  applyThreePassMaterial(first, material({stencilWrite: true, stencilRef: 19, stencilWriteMask: 0x42,
    stencilFunc: T.EqualStencilFunc, stencilZPass: T.ReplaceStencilOp}), raster);
  finishThreePassState(first);
  const skippedFinal = finishThreePassState(cancelled, {render: false});
  assert.equal(skippedFinal.colorMask, false); assert.equal(skippedFinal.depthMask, false);
  const values = beginThreePassState(next);
  assert.equal(values.stencilFunc, T.NotEqualStencilFunc); assert.equal(values.stencilRef, 7);
  assert.equal(values.stencilFuncMask, 0x21); assert.equal(values.stencilWriteMask, 0x42);
  assert.equal(values.stencilZPass, T.ReplaceStencilOp); assert.equal(values.colorMask, false);
  finishThreePassState(next);
});

test('consecutive cancelled tokens finalize in order and active failures preserve executed state', () => {
  const {state, capture} = fixture(), first = capture();
  state.buffers.stencil.setTest(true);
  const second = capture();
  state.buffers.stencil.setMask(0x19);
  const third = capture();
  state.buffers.stencil.setLocked(true);
  const last = capture();
  finishThreePassState(third, {render: false}); finishThreePassState(second, {render: false});
  applyThreePassMaterial(first, material({colorWrite: false, depthWrite: false, stencilWrite: true, stencilRef: 11}), raster);
  const failed = finishThreePassState(first, {render: false});
  assert.equal(failed.colorMask, false); assert.equal(failed.depthMask, false);
  assert.equal(finishThreePassState(second, {render: false}).stencilWriteMask, 0xff);
  assert.equal(finishThreePassState(third, {render: false}).stencilWriteMask, 0x19);
  const values = beginThreePassState(last);
  assert.equal(values.colorMask, false); assert.equal(values.depthMask, false);
  assert.equal(values.stencilRef, 11); assert.equal(values.stencilWriteMask, 0x19); assert.equal(values.stencilTest, true);
  finishThreePassState(last);
});

test('first-token cancellation preserves the live ordinary baseline and folds repeated rejected requests', () => {
  const renderer = {};
  registerThreePassState(renderer, T);
  assert.equal(captureThreePassState(renderer), null);
  const state = getThreePassState(renderer);
  state.buffers.stencil.setLocked(true);
  const first = captureThreePassState(renderer);
  assert.equal(finishThreePassState(first, {render: false}), null);
  for (let index = 0; index < 256; index++) {
    state.buffers.stencil.setClear(index);
    assert.equal(finishThreePassState(captureThreePassState(renderer), {render: false}), null);
  }
  const next = captureThreePassState(renderer);
  recordThreePassMaterial(renderer, material({stencilWrite: true, stencilFunc: T.EqualStencilFunc,
    stencilRef: 41, stencilWriteMask: 0x72, stencilZPass: T.ReplaceStencilOp}));
  const values = beginThreePassState(next);
  assert.equal(values.stencilRef, 41); assert.equal(values.stencilWriteMask, 0x72);
  assert.equal(values.stencilFunc, T.EqualStencilFunc); assert.equal(values.stencilZPass, T.ReplaceStencilOp);
  assert.equal(values.stencilLocked, true); assert.equal(values.stencilClear, 255);
  const result = applyThreePassMaterial(next, material(), raster);
  assert.equal(result.stencilReference, 41); assert.equal(result.stencil.writeMask, 0x72);
  finishThreePassState(next);
});

test('clear resets unlocked depth/stencil masks, preserves color masking and converts stencil to eight bits', () => {
  const {state, capture} = fixture();
  state.buffers.color.setMask(false); state.buffers.depth.setMask(false); state.buffers.stencil.setMask(0);
  state.buffers.stencil.setClear(-257);
  const token = capture(), cleared = applyThreePassClear(token, frame);
  assert.equal(cleared.loadOp, 'load'); assert.equal(cleared.depthLoadOp, 'clear'); assert.equal(cleared.stencilLoadOp, 'clear');
  assert.equal(cleared.clearStencil, 255); assert.equal(cleared.clearColor, frame.clearColor);
  const final = finishThreePassState(token, {render: false});
  assert.equal(final.colorMask, false); assert.equal(final.depthMask, true); assert.equal(final.stencilWriteMask, 0xffffffff);
  assert.equal(frame.loadOp, 'clear'); assert.equal(frame.clearStencil, 0);
});

test('clear retains locked color/depth and zero-write stencil contents', () => {
  const {state, capture} = fixture();
  state.buffers.color.setMask(false); state.buffers.color.setLocked(true);
  state.buffers.depth.setMask(false); state.buffers.depth.setLocked(true);
  state.buffers.stencil.setMask(0); state.buffers.stencil.setLocked(true); state.buffers.stencil.setClear(4);
  const token = capture(), cleared = applyThreePassClear(token, frame);
  assert.equal(cleared.loadOp, 'load'); assert.equal(cleared.depthLoadOp, 'load'); assert.equal(cleared.stencilLoadOp, 'load');
  assert.equal(cleared.clearStencil, 4); finishThreePassState(token, {render: false});
});

test('requested clears reset unlocked state even without attachments, and false requests preserve masks', () => {
  const {state, capture} = fixture();
  state.buffers.depth.setMask(false); state.buffers.stencil.setMask(0);
  const first = capture(), noAttachments = Object.freeze({loadOp: 'load'});
  assert.deepEqual(applyThreePassClear(first, noAttachments, {color: false, depth: true, stencil: true}), noAttachments);
  const final = finishThreePassState(first, {render: false});
  assert.equal(final.depthMask, true); assert.equal(final.stencilWriteMask, 0xffffffff);
  state.buffers.depth.setMask(false); state.buffers.stencil.setMask(0);
  const second = capture(), retained = applyThreePassClear(second, frame, {color: true, depth: false, stencil: false});
  assert.equal(retained.loadOp, 'clear'); assert.equal(retained.depthLoadOp, 'load'); assert.equal(retained.stencilLoadOp, 'load');
  const retainedFinal = finishThreePassState(second, {render: false});
  assert.equal(retainedFinal.depthMask, false); assert.equal(retainedFinal.stencilWriteMask, 0);
});

test('a partial locked stencil clear never silently overwrites unmasked bits', () => {
  const {state, capture} = fixture();
  state.buffers.stencil.setMask(0x0f); state.buffers.stencil.setLocked(true);
  const token = capture();
  assert.throws(() => applyThreePassClear(token, frame), {code: 'THREE_PASS_STATE_UNSUPPORTED'});
  finishThreePassState(token, {render: false});
});

test('renderer instances keep separate explicit writes and ordered material histories', () => {
  const a = fixture(), b = fixture();
  a.state.buffers.stencil.setTest(true); a.state.buffers.stencil.setLocked(true);
  a.state.buffers.color.setMask(false); a.state.buffers.color.setLocked(true);
  const first = a.capture(), other = b.capture();
  assert.equal(applyThreePassMaterial(first, material(), raster).writeMask, 0);
  assert.equal(applyThreePassMaterial(other, material(), raster), raster);
  finishThreePassState(first); finishThreePassState(other);
});
