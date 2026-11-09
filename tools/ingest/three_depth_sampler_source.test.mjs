/** Execute the pinned WebGLTextures owner; the GL boundary records image and
 * parameter identity only. No source behavior is reimplemented in the test. */
import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';

const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
const {WebGLTextures} = await import(pathToFileURL(path.join(root, 'src/renderers/webgl/WebGLTextures.js')));
const {WebGLProperties} = await import(pathToFileURL(path.join(root, 'src/renderers/webgl/WebGLProperties.js')));

function sourceOwner() {
  let bound, next = 1;
  const images = [], properties = WebGLProperties();
  const gl = new Proxy({TEXTURE0: 0,
    createTexture() {const image = {parameters: new Map()}; images.push(image); return image;},
    deleteTexture(image) {image.destroyed = true;},
    texParameteri(_target, key, value) {bound.parameters.set(key, value);},
    createFramebuffer() {return {};}, deleteFramebuffer() {}, framebufferTexture2D() {},
  }, {get(object, key) {if (!(key in object)) object[key] = next++; return object[key];}});
  const state = {activeTexture() {}, bindTexture(_target, image) {bound = image;}, unbindTexture() {bound = null;},
    pixelStorei() {}, bindFramebuffer() {},
    texImage2D(_target, _level, format, width, height) {Object.assign(bound, {format, width, height});},
    texStorage2D(_target, _levels, format, width, height) {Object.assign(bound, {format, width, height});}};
  const convert = value => value === T.RGBAFormat ? gl.RGBA : value === T.DepthFormat ? gl.DEPTH_COMPONENT
    : value === T.UnsignedByteType ? gl.UNSIGNED_BYTE : value === T.FloatType ? gl.FLOAT : value;
  const owner = new WebGLTextures(gl, {has: name => name === 'OES_texture_float_linear'}, state, properties,
    {maxTextures: 16, maxTextureSize: 8192, maxSamples: 4}, {convert}, {memory: {textures: 0}});
  return {owner, gl, images, image: texture => properties.get(texture).__webglTexture};
}

test('source attached depth comparison updates require versions and null keeps the existing native comparison', () => {
  const f = sourceOwner(), depth = new T.DepthTexture(4, 4, T.FloatType);
  depth.compareFunction = T.LessEqualCompare;
  const target = new T.WebGLRenderTarget(4, 4, {depthTexture: depth});
  try {
    assert.equal(depth.version, 0);
    f.owner.setupRenderTarget(target);
    assert.equal(depth.version, 1, 'source target initialization requests the first depth upload');
    const image = f.image(depth), parameters = image.parameters;
    assert.equal(parameters.get(f.gl.TEXTURE_COMPARE_MODE), f.gl.COMPARE_REF_TO_TEXTURE);
    assert.equal(parameters.get(f.gl.TEXTURE_COMPARE_FUNC), f.gl.LEQUAL);
    depth.compareFunction = T.GreaterCompare; f.owner.setTexture2D(depth, 0);
    assert.equal(f.image(depth), image);
    assert.equal(parameters.get(f.gl.TEXTURE_COMPARE_FUNC), f.gl.LEQUAL, 'public edits without upload leave the installed comparator');
    depth.needsUpdate = true; f.owner.setTexture2D(depth, 0);
    assert.equal(f.image(depth), image, 'comparison changes do not change the pinned texture cache key');
    assert.equal(parameters.get(f.gl.TEXTURE_COMPARE_FUNC), f.gl.GREATER);
    depth.compareFunction = null; depth.needsUpdate = true; f.owner.setTexture2D(depth, 0);
    assert.equal(f.image(depth), image);
    assert.equal(parameters.get(f.gl.TEXTURE_COMPARE_MODE), f.gl.COMPARE_REF_TO_TEXTURE);
    assert.equal(parameters.get(f.gl.TEXTURE_COMPARE_FUNC), f.gl.GREATER, 'source does not write NONE when comparison is removed');
    depth.wrapS = T.RepeatWrapping; f.owner.setTexture2D(depth, 0);
    assert.equal(f.image(depth), image, 'filter/wrap cache changes are also gated by needsUpdate');
    depth.needsUpdate = true; f.owner.setTexture2D(depth, 0);
    const replaced = f.image(depth);
    assert.notEqual(replaced, image); assert.equal(image.destroyed, true);
    assert.equal(replaced.parameters.get(f.gl.TEXTURE_WRAP_S), f.gl.REPEAT);
    assert.equal(replaced.parameters.has(f.gl.TEXTURE_COMPARE_MODE), false, 'replacement storage starts with default comparison disabled');
    assert.equal(replaced.parameters.has(f.gl.TEXTURE_COMPARE_FUNC), false);
    const oldCount = f.images.length;
    depth.magFilter = T.LinearFilter; depth.needsUpdate = true; f.owner.setTexture2D(depth, 0);
    assert.equal(f.images.length, oldCount + 1); assert.equal(replaced.destroyed, true);
    assert.equal(f.image(depth).parameters.get(f.gl.TEXTURE_MAG_FILTER), f.gl.LINEAR);
  } finally {target.dispose();}
});
