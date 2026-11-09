import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createThreePassSnapshots} from './three_pass_snapshot.mjs';
import {createThreeProgramSupport} from './three_program.mjs';
const T = await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js'), 'build/three.module.js')));

function fixture(t) {
  const scalar = new T.Uniform(7), vector = new T.Uniform(new T.Vector3(1, 2, 3));
  const group = new T.UniformsGroup().setName('Controls').add(scalar).add(vector);
  const material = new T.RawShaderMaterial({glslVersion: T.GLSL3, uniformsGroups: [group],
    uniforms: {gain: new T.Uniform(3)},
    vertexShader: 'precision highp float; in vec3 position; void main() { gl_Position = vec4(position, 1.); }',
    fragmentShader: 'precision highp float; uniform float gain; layout(std140) uniform Controls { float factor; vec3 vector; } controls;' +
      'out vec4 color; void main() { color = vec4(controls.vector * gain, controls.factor); }'});
  const mesh = new T.Mesh(new T.PlaneGeometry(2, 2), material), camera = new T.OrthographicCamera();
  const snapshots = createThreePassSnapshots(T);
  const support = createThreeProgramSupport({three: T, state: () => ({toneMapping: T.NoToneMapping, outputColorSpace: T.LinearSRGBColorSpace})});
  function pack(current = mesh, nextFrame = true) {
    if (nextFrame) support.beginFrame();
    const compiled = support.compile(current.material, current), {reflection} = compiled.program;
    const bytes = new Uint8Array(reflection.uniformBufferSize), view = new DataView(bytes.buffer);
    support.pack(reflection, support.refresh(current.material), current, camera, bytes, {material: current.material});
    const read = (name, count = 1) => {
      const uniform = reflection.uniforms.find(uniform => uniform.name === name);
      assert.ok(uniform, name);
      return Array.from({length: count}, (_, i) => view.getFloat32(uniform.offset + i * 4, true));
    };
    return {compiled, gain: read('gain')[0], factor: read('controls_factor')[0], vector: read('controls_vector', 3)};
  }
  t.after(() => { snapshots.dispose(); support.dispose(); group.dispose(); material.dispose(); mesh.geometry.dispose(); });
  return {scalar, vector, group, material, mesh, camera, snapshots, support, pack};
}

test('fullscreen program caches observe each accepted ordinary and block uniform set', t => {
  const f = fixture(t), first = f.snapshots.capture(f.mesh, f.camera);
  f.scalar.value = 11; f.vector.value.set(4, 5, 6); f.material.uniforms.gain.value = 5;
  const second = f.snapshots.capture(f.mesh, f.camera);
  f.scalar.value = 99; f.vector.value.set(99, 99, 99); f.material.uniforms.gain.value = 99;
  first.install(); const uniforms = first.root.material.uniforms, a = f.pack(first.root); first.release();
  second.install(); const b = f.pack(second.root); second.release();
  assert.equal(second.root.material.uniforms, uniforms, 'the linked program keeps its uniform dictionary');
  assert.equal(a.compiled, b.compiled, 'changing values does not compile another shader');
  assert.deepEqual([a.gain, a.factor, a.vector], [3, 7, [1, 2, 3]]);
  assert.deepEqual([b.gain, b.factor, b.vector], [5, 11, [4, 5, 6]]);
  assert.equal(f.material.uniforms.gain.value, 99); assert.equal(f.scalar.value, 99);
  assert.equal(f.group.__size, 32, 'private captures use the source block layout owner');
});

test('live and captured materials share the source group once-per-frame upload history', t => {
  const f = fixture(t); assert.equal(f.pack().factor, 7);
  f.scalar.value = 11;
  const capture = f.snapshots.capture(f.mesh, f.camera); capture.install(); f.scalar.value = 99;
  assert.equal(f.pack(capture.root, false).factor, 7, 'later uses in the same frame keep the first upload');
  assert.equal(f.pack(capture.root).factor, 11, 'the next frame consumes the captured input');
  assert.equal(f.pack(f.mesh, false).factor, 11, 'a live material observes that same uploaded generation');
  assert.equal(f.pack().factor, 99); capture.release();
});

test('queued block captures retain disposal generations and rebuild reused layouts', t => {
  const f = fixture(t), first = f.snapshots.capture(f.mesh, f.camera);
  f.group.dispose();
  const data = new Float32Array([13, 0, 0, 0, 17, 19, 23, 0]);
  f.group.uniforms = [new T.Uniform(data)];
  const second = f.snapshots.capture(f.mesh, f.camera);
  f.scalar.value = 99; f.vector.value.set(99, 99, 99); data.fill(99);
  first.install(); const a = f.pack(first.root); first.release();
  second.install(); const b = f.pack(second.root); second.release();
  assert.deepEqual([a.factor, a.vector], [7, [1, 2, 3]]);
  assert.deepEqual([b.factor, b.vector], [13, [17, 19, 23]]);
  assert.equal(f.group.uniforms[0].__offset, 0);
  assert.equal(f.group.uniforms[0].__data.length, 8, 'disposal resets the source layout for the new payload');
});

test('uniform block payloads obey the existing snapshot budget and release with their owner', t => {
  const f = fixture(t), bounded = createThreePassSnapshots(T, {maxCaptureBytes: 1024});
  t.after(() => bounded.dispose());
  f.group.uniforms = [new T.Uniform(new Float32Array(1024))];
  assert.throws(() => bounded.capture(f.mesh, f.camera), {code: 'THREE_PASS_BUDGET'});
  assert.equal(bounded.allocatedBytes, 0);
  const capture = f.snapshots.capture(f.mesh, f.camera); capture.install(); capture.release();
  assert.ok(f.snapshots.allocatedBytes > 0);
  f.snapshots.dispose(); assert.equal(f.snapshots.allocatedBytes, 0);
});
