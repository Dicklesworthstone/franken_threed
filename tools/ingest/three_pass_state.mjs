/** Private r186 postprocessing state. Retained passes can select buffer locks
 * without exposing a fabricated WebGL context or renderer.state surface.
 * Explicit pass calls are captured at acceptance; material changes resolve in
 * submitted order, after the preceding render has supplied its final state.
 */
import {threeProgramStencil} from './three_program_stencil.mjs';

const renderers = new WeakMap(), tokens = new WeakMap();
const STENCIL_FIELDS = ['stencilFunc', 'stencilRef', 'stencilFuncMask', 'stencilWriteMask',
  'stencilFail', 'stencilZFail', 'stencilZPass'];
const FACE_FIELDS = ['compare', 'failOp', 'depthFailOp', 'passOp'];
const EMPTY_STENCIL = Object.freeze({});
const fail = (code, message) => {
  throw Object.assign(new Error(`THREE_PASS_STATE_${code}: ${message}`), {name: 'ThreePassStateError', code: `THREE_PASS_STATE_${code}`});
};

export function registerThreePassState(renderer, three) {
  const existing = renderers.get(renderer);
  if (existing) return existing.state;
  const initial = {colorMask: true, colorLocked: false, depthMask: true, depthLocked: false,
    stencilTest: false, stencilLocked: false, stencilWriteMask: 0xffffffff,
    stencilFunc: three.AlwaysStencilFunc, stencilRef: 0, stencilFuncMask: 0xffffffff,
    stencilFail: three.KeepStencilOp, stencilZFail: three.KeepStencilOp, stencilZPass: three.KeepStencilOp, stencilClear: 0};
  const record = {three, targets: renderer._targets, initial, active: false, latest: null, pending: {}, locks: {color: false, depth: false, stencil: false}};
  const write = (key, value) => { record.pending[key] = value; };
  const setLocked = name => value => {
    record.locks[name] = !!value; write(`${name}Locked`, !!value);
  };
  const buffer = name => Object.freeze({
    setMask(value) { if (!record.locks[name]) write(`${name}Mask`, !!value); },
    setLocked: setLocked(name),
  });
  record.state = Object.freeze({buffers: Object.freeze({color: buffer('color'), depth: buffer('depth'),
    stencil: Object.freeze({
      setTest(value) { if (!record.locks.stencil) write('stencilTest', !!value); },
      setMask(value) { if (!record.locks.stencil) write('stencilWriteMask', value >>> 0); },
      setLocked: setLocked('stencil'),
      setFunc(func, reference, mask) {
        write('stencilFunc', func >>> 0); write('stencilRef', reference | 0); write('stencilFuncMask', mask >>> 0);
      },
      setOp(fail, depthFail, pass) {
        write('stencilFail', fail >>> 0); write('stencilZFail', depthFail >>> 0); write('stencilZPass', pass >>> 0);
      },
      setClear(value) { write('stencilClear', value | 0); },
    }),
  })});
  record.constants = Object.freeze({NEVER: three.NeverStencilFunc, LESS: three.LessStencilFunc,
    EQUAL: three.EqualStencilFunc, LEQUAL: three.LessEqualStencilFunc, GREATER: three.GreaterStencilFunc,
    NOTEQUAL: three.NotEqualStencilFunc, GEQUAL: three.GreaterEqualStencilFunc, ALWAYS: three.AlwaysStencilFunc,
    ZERO: three.ZeroStencilOp, KEEP: three.KeepStencilOp, REPLACE: three.ReplaceStencilOp,
    INCR: three.IncrementStencilOp, DECR: three.DecrementStencilOp, INCR_WRAP: three.IncrementWrapStencilOp,
    DECR_WRAP: three.DecrementWrapStencilOp, INVERT: three.InvertStencilOp});
  renderers.set(renderer, record);
  return record.state;
}

export function getThreePassState(renderer) {
  const record = renderers.get(renderer);
  if (!record) return renderer.state;
  record.active = true; return record.state;
}
export function getThreeStencilConstants(renderer) {
  const record = renderers.get(renderer);
  if (!record) return renderer.getContext();
  record.active = true; return record.constants;
}

/** Source-verified supersampling passes temporarily change their camera's
 * projection for each render call. The owned renderer must capture that call
 * before the pass restores the public camera. Exact renderers execute normally;
 * no public state flag or camera replacement is installed on either renderer.
 */
export function renderThreePassScene(renderer, scene, camera) {
  const record = renderers.get(renderer);
  if (!record) return renderer.render(scene, camera);
  const previous = record.captureCamera;
  record.captureCamera = true;
  try { return renderer.render(scene, camera); }
  finally { record.captureCamera = previous; }
}

export function requiresThreePassCamera(renderer) {
  return renderers.get(renderer)?.captureCamera === true;
}

/** The verified composer calls its original setSize body inside this scope.
 * Its own targets and the standard passes' direct target fields are finite;
 * inspect data descriptors only, never application getters or nested metadata.
 * Actual disposal and unscoped resize retain their normal invalidation rules.
 */
export function withThreeComposerResize(composer, run) {
  const renderer = Object.getOwnPropertyDescriptor(composer, 'renderer')?.value;
  const record = renderers.get(renderer);
  if (!record) return run();
  const targets = new Set();
  const add = value => {
    if (!(value instanceof record.three.RenderTarget)) return;
    targets.add(value);
    if (targets.size > 64) fail('LIMIT', 'Composer resize exceeds the retained target capacity');
  };
  const fields = value => {
    for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(value))) {
      if (!Object.hasOwn(descriptor, 'value')) continue;
      const entry = descriptor.value;
      if (!Array.isArray(entry)) { add(entry); continue; }
      const elements = Object.getOwnPropertyDescriptors(entry);
      if (elements.length.value > 64) fail('LIMIT', 'Composer resize target arrays exceed capacity');
      for (const [key, element] of Object.entries(elements))
        if (/^(0|[1-9][0-9]*)$/.test(key) && Object.hasOwn(element, 'value')) add(element.value);
    }
  };
  add(Object.getOwnPropertyDescriptor(composer, 'renderTarget1')?.value);
  add(Object.getOwnPropertyDescriptor(composer, 'renderTarget2')?.value);
  const passes = Object.getOwnPropertyDescriptor(composer, 'passes')?.value;
  if (!Array.isArray(passes) || passes.length > 64) fail('LIMIT', 'Expected a finite source composer pass array');
  for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(passes))) {
    if (!/^(0|[1-9][0-9]*)$/.test(key)) continue;
    if (!Object.hasOwn(descriptor, 'value') || !descriptor.value) fail('SOURCE', 'Expected source composer pass identities');
    fields(descriptor.value);
  }
  return targets.size ? record.targets.withNativeEffectTargets([...targets], run) : run();
}

// Requests accepted before a retained pass activates this adapter have no
// token. Their executed stencil state still precedes the first private token.
// These hooks retain that baseline without preparing raster variants, changing
// accepted explicit writes, or imposing private state work on ordinary draws.
export function recordThreePassMaterial(renderer, material) {
  const record = renderers.get(renderer);
  if (!record) return;
  record.initial.stencilTest = !!material.stencilWrite;
  if (material.stencilWrite) for (const field of STENCIL_FIELDS) record.initial[field] = material[field];
}
export function recordThreePassClear(renderer, frame, {stencil = frame.stencilLoadOp === 'clear'} = {}) {
  const record = renderers.get(renderer);
  if (record && stencil) record.initial.stencilWriteMask = 0xffffffff;
}

export function captureThreePassState(renderer) {
  const record = renderers.get(renderer);
  if (!record?.active) return null;
  const token = Object.freeze({});
  tokens.set(token, {record, initial: record.initial, parent: record.latest, skipped: false, changes: record.pending, working: null, final: null,
    stencilValues: null, stencilRaster: EMPTY_STENCIL});
  record.pending = {}; record.latest = token;
  return token;
}
function captured(token) {
  const capture = tokens.get(token);
  if (!capture) fail('TOKEN', 'Expected a captured private pass state');
  return capture;
}
function workingState(capture) {
  if (capture.working) return capture.working;
  const previous = capture.parent ? captured(capture.parent).final : capture.initial;
  if (!previous) fail('ORDER', 'Previous pass state must finish before this request executes');
  capture.working = {...previous, ...capture.changes}; capture.changes = null;
  return capture.working;
}
function finalize(capture) {
  capture.final = Object.freeze({...workingState(capture)});
  capture.parent = capture.changes = capture.working = capture.record = capture.initial = capture.stencilValues = null;
  capture.stencilRaster = EMPTY_STENCIL;
  return capture.final;
}
function resolveSkippedParents(capture) {
  const skipped = [];
  let previous = capture.parent && captured(capture.parent);
  while (previous && !previous.final && previous.skipped) {
    skipped.push(previous); previous = previous.parent && captured(previous.parent);
  }
  if (previous && !previous.final) fail('ORDER', 'Previous pass state must finish before this request executes');
  for (let index = skipped.length - 1; index >= 0; index--) finalize(skipped[index]);
}
export function beginThreePassState(token) {
  if (token == null) return null;
  const capture = captured(token);
  if (capture.final || capture.skipped) fail('ORDER', 'A completed or cancelled pass state cannot execute again');
  if (!capture.working) resolveSkippedParents(capture);
  return workingState(capture);
}
function sameStencil(a, b) {
  if (a == null || b == null) return a == null && b == null;
  return a.readMask === b.readMask && a.writeMask === b.writeMask &&
    ['front', 'back'].every(face => FACE_FIELDS.every(field => a[face]?.[field] === b[face]?.[field]));
}
export function applyThreePassMaterial(token, material, baseRaster) {
  if (token == null) return baseRaster;
  const state = beginThreePassState(token), capture = captured(token);
  if (!state.colorLocked) state.colorMask = !!material.colorWrite;
  if (!state.depthLocked) state.depthMask = !!material.depthWrite;
  if (!state.stencilLocked) state.stencilTest = !!material.stencilWrite;
  // WebGLState locks only the stencil test and write mask. A material that
  // writes stencil still changes comparison, reference and operations.
  if (material.stencilWrite) {
    if (!state.stencilLocked) state.stencilWriteMask = material.stencilWriteMask >>> 0;
    for (const field of STENCIL_FIELDS) if (field !== 'stencilWriteMask') state[field] = material[field];
  }
  let mapped = EMPTY_STENCIL;
  if (state.stencilTest) {
    const values = STENCIL_FIELDS.map(field => state[field]);
    if (!capture.stencilValues || values.some((value, index) => value !== capture.stencilValues[index])) {
      capture.stencilRaster = threeProgramStencil(capture.record.three, {...state, stencilWrite: true}, fail);
      capture.stencilValues = values;
    }
    mapped = capture.stencilRaster;
  }
  // DEPTH_TEST suppresses GL fragment writes independently of the depth mask.
  // Preserve that mask itself: a later clear or tested draw still consumes it.
  const writeMask = state.colorMask ? 0xf : 0, depthWriteEnabled = !!material.depthTest && state.depthMask;
  if (baseRaster.writeMask === writeMask && baseRaster.depthWriteEnabled === depthWriteEnabled &&
      baseRaster.stencilReference === mapped.stencilReference && sameStencil(baseRaster.stencil, mapped.stencil)) return baseRaster;
  const {stencil, stencilReference, ...base} = baseRaster;
  return Object.freeze({...base, writeMask, depthWriteEnabled, ...mapped});
}

export function applyThreePassClear(token, frame, {depth = frame.depthLoadOp === 'clear', stencil = frame.stencilLoadOp === 'clear'} = {}) {
  if (token == null) return frame;
  const state = beginThreePassState(token), result = {...frame};
  if (frame.loadOp === 'clear' && !state.colorMask) result.loadOp = 'load';
  if (depth && !state.depthLocked) state.depthMask = true;
  if (stencil && !state.stencilLocked) state.stencilWriteMask = 0xffffffff;
  if (frame.depthLoadOp === 'clear') {
    if (!state.depthMask) result.depthLoadOp = 'load';
  }
  if (frame.stencilLoadOp === 'clear') {
    const mask = state.stencilWriteMask & 0xff;
    if (mask === 0) result.stencilLoadOp = 'load';
    else if (mask !== 0xff) fail('UNSUPPORTED', 'A locked partial stencil clear requires a masked clear draw');
    result.clearStencil = state.stencilClear & 0xff;
  }
  return result;
}

export function finishThreePassState(token, {render = true} = {}) {
  if (token == null) return null;
  const capture = captured(token);
  if (capture.final) return capture.final;
  if (capture.skipped) render = false;
  if (!capture.working && !render) {
    // A rejected first private request can precede ordinary queued draws. Keep
    // its baseline live until an ordered successor executes. Fold consecutive
    // cancellations so repeated queue-limit failures retain no token history.
    while (capture.parent) {
      const previous = captured(capture.parent);
      if (!previous.skipped || previous.final) break;
      capture.changes = {...previous.changes, ...capture.changes};
      capture.parent = previous.parent;
    }
    capture.skipped = true;
    capture.record = null;
    if (!capture.parent || !captured(capture.parent).final) return null;
  }
  if (!capture.working) resolveSkippedParents(capture);
  const state = workingState(capture);
  if (render) {
    if (!state.colorLocked) state.colorMask = true;
    if (!state.depthLocked) state.depthMask = true;
  }
  return finalize(capture);
}
