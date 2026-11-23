/** Resource-ordering guards for native object-callback compute. These tests
 * exercise the actual capture module, without a GPU or shader arithmetic. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {withThreeNativeCapture, markThreeNativeDrawCaptured, withThreeNativeObjectHooks, withThreeNativeComputePrerequisites, assertThreeNativeTargetWrite, assertThreeNativeTextureWrite, assertThreeNativeComputeOutsideCapture} from './three_node_shadows.mjs';

const callbackError = {code: 'THREE_NODE_PROGRAM_REENTRY'};
const computeError = {code: 'THREE_COMPUTE_REENTRY'};

test('object callback compute can produce resources before the first parent draw', () => {
  const renderer = {}, produced = {};
  withThreeNativeCapture(renderer, () => withThreeNativeObjectHooks(renderer, () => {
    assert.equal(assertThreeNativeComputeOutsideCapture(renderer), true);
    assert.equal(assertThreeNativeComputeOutsideCapture(renderer, [produced]), true);
  }));
});

test('later object callbacks can compute resources disjoint from earlier parent reads', () => {
  const renderer = {}, sampled = {}, uploaded = {}, output = {};
  withThreeNativeCapture(renderer, (texture, buffer) => {
    texture(sampled); buffer(uploaded); markThreeNativeDrawCaptured(renderer, true);
    withThreeNativeObjectHooks(renderer, () =>
      assert.equal(assertThreeNativeComputeOutsideCapture(renderer, [output]), true));
  });
});

for (const kind of ['buffer', 'texture'])
test(`callback compute cannot borrow the geometry identity exemption for a prior ${kind}`, () => {
  const renderer = {}, resource = {}, identity = {};
  withThreeNativeCapture(renderer, (texture, buffer) => {
    (kind === 'buffer' ? buffer : texture)(resource, identity);
    markThreeNativeDrawCaptured(renderer, true);
    withThreeNativeObjectHooks(renderer, () =>
      assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [resource],
        () => identity, () => identity), callbackError));
  });
});

for (const kind of ['color', 'depth'])
test(`callback compute refuses the active ${kind} attachment before the first draw`, () => {
  const renderer = {}, attachment = {};
  const target = kind === 'color' ? {textures: [attachment]} : {depthTexture: attachment};
  withThreeNativeCapture(renderer, () => withThreeNativeObjectHooks(renderer, () =>
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [attachment]), callbackError)), target);
});

test('an unknown earlier draw does not grant callback compute permission', () => {
  const renderer = {};
  withThreeNativeCapture(renderer, () => {
    markThreeNativeDrawCaptured(renderer);
    withThreeNativeObjectHooks(renderer, () =>
      assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [{}]), callbackError));
  });
});

test('ordinary compute still needs an explicit prerequisite or object callback boundary', () => {
  const renderer = {};
  withThreeNativeCapture(renderer, () =>
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [{}]), computeError));
});

test('registered geometry prerequisites retain their unchanged buffer and texture identities', () => {
  const renderer = {}, attribute = {}, image = {}, upload = {}, residency = {};
  withThreeNativeCapture(renderer, (texture, buffer) => {
    buffer(attribute, upload); texture(image, residency); markThreeNativeDrawCaptured(renderer, true);
    withThreeNativeComputePrerequisites(renderer, () =>
      assert.equal(assertThreeNativeComputeOutsideCapture(renderer, [attribute, image],
        () => upload, () => residency), true), true);
  });
});

test('changed upload identities still invalidate registered geometry prerequisites', () => {
  const renderer = {}, attribute = {};
  withThreeNativeCapture(renderer, (_texture, buffer) => {
    buffer(attribute, {}); markThreeNativeDrawCaptured(renderer, true);
    withThreeNativeComputePrerequisites(renderer, () =>
      assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [attribute], () => ({})), computeError), true);
  });
});

test('a nested callback checks every pending ancestor for conflicting reads', () => {
  const renderer = {}, ancestorBuffer = {};
  withThreeNativeCapture(renderer, (_texture, buffer) => {
    buffer(ancestorBuffer); markThreeNativeDrawCaptured(renderer, true);
    withThreeNativeCapture(renderer, () => withThreeNativeObjectHooks(renderer, () =>
      assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [ancestorBuffer]), callbackError)));
  });
});

test('deferred compute compilation remains refused inside an object callback', () => {
  const renderer = {};
  withThreeNativeCapture(renderer, () => withThreeNativeObjectHooks(renderer, () =>
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, null, null, null, false), callbackError)));
});

test('object callback compute permission does not authorize nested texture or target writes', () => {
  const renderer = {}, texture = {};
  withThreeNativeCapture(renderer, () => withThreeNativeObjectHooks(renderer, () => {
    assert.throws(() => assertThreeNativeTextureWrite(renderer, texture), callbackError);
    assert.throws(() => assertThreeNativeTargetWrite(renderer, {textures: [texture]}), callbackError);
  }));
});

test('resource revalidation rejects a late overlapping binding before queue publication', () => {
  const renderer = {}, resource = {};
  withThreeNativeCapture(renderer, (_texture, buffer) => withThreeNativeObjectHooks(renderer, () => {
    assert.equal(assertThreeNativeComputeOutsideCapture(renderer, [resource]), true);
    buffer(resource); markThreeNativeDrawCaptured(renderer, true);
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [resource]), callbackError);
  }));
});

test('capture and callback state are restored when source callbacks throw', () => {
  const renderer = {}, marker = new Error('source callback');
  assert.throws(() => withThreeNativeCapture(renderer, () => withThreeNativeObjectHooks(renderer, () => {
    assert.equal(assertThreeNativeComputeOutsideCapture(renderer, [{}]), true); throw marker;
  })), error => error === marker);
  assert.equal(assertThreeNativeComputeOutsideCapture(renderer, [{}]), false);
  assert.equal(assertThreeNativeComputeOutsideCapture(renderer, null, null, null, false), false);
  assert.doesNotThrow(() => assertThreeNativeTextureWrite(renderer, {}));
});

test('a temporarily inactive callback does not leak its compute permission into the draw handler', () => {
  const renderer = {};
  withThreeNativeCapture(renderer, () => withThreeNativeObjectHooks(renderer, () => {
    withThreeNativeObjectHooks(renderer, () =>
      assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer, [{}]), computeError), false);
    assert.equal(assertThreeNativeComputeOutsideCapture(renderer, [{}]), true);
  }));
});
