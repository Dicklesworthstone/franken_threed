/** Pinned r186 standard-pass ownership and the small stencil-state lowering.
 * Source identity and resolved dependencies authorize this transform, never a
 * filename. Custom passes and changed implementations retain exact ownership. */
import {createHash} from 'node:crypto';

const pins = new Map([
  ['Pass.js', ['444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5', {Pass: 'base'}]],
  ['EffectComposer.js', ['4e079a5886152d7e529a59aef644e968ab4d32c6a33ce016b36bf29b2eac26f7', {EffectComposer: 'composer'}]],
  ['MaskPass.js', ['7cd08eee9d5d6f5578beaddbdcbe9c384f6873810af27f22ab7db3ceeb127aa3', {MaskPass: 'pass', ClearMaskPass: 'pass'}]],
  ['RenderPass.js', ['817f6c3cdcd0fd41515d112359ea0532568eefb5aabd3b33903957ebca1b8a6a', {RenderPass: 'pass'}]],
  ['ShaderPass.js', ['e2500a5913b26bbf5148ceaae644c6edcff06a18b01494ee37bf856353d2ab9d', {ShaderPass: 'pass'}]],
  ['BloomPass.js', ['5dcc005e8782003124a3e1406bbc37ce9b014791e263692fb098a81fc9c45e2a', {BloomPass: 'pass'}]],
  ['FilmPass.js', ['709a6e640b5ced2449e458d57656173208e81af064e414b9c6f775cbf1287ed2', {FilmPass: 'pass'}]],
  ['DotScreenPass.js', ['ef8fc81f6697a76abf42fe8a5e16343c7254b8ece68f2f2eb7a20190c86e3fe1', {DotScreenPass: 'pass'}]],
  ['TexturePass.js', ['402cfcdda8e69db5f0faa778c51b97fc2ec740846a088d169f555fc3403b7d7d', {TexturePass: 'pass'}]],
  ['ClearPass.js', ['69c525e8e3be28999453177a9d0bd1771d27a4d04378307242802fec84301bcb', {ClearPass: 'pass'}]],
  ['OutputPass.js', ['02e4a261af34de71338185e9e87f0cbe5cba9115608d984363e1269dec1d2272', {OutputPass: 'pass'}]],
  ['CopyShader.js', ['a33057d5ac91c43304c186ac0e8816e62bb2ed471d3a00ff3018dfd5c0389718', {}]],
  ['ConvolutionShader.js', ['bad71ed02351acb701e62e21a9d11f044c2c689d924d714061edb756fbea8ace', {}]],
  ['FilmShader.js', ['744c619c70f949af5d3f0f9edb27573743ab9ad12869e7c306ffb3d0caff7bb5', {}]],
  ['DotScreenShader.js', ['959a2e0ea3d4a89d67754e417c56bd67abd632427d51fab19361003cd0ca7cef', {}]],
  ['OutputShader.js', ['353479f77a8d7e2629d49ccac9fc2f5dbfdda5442e0adf867b00377a2fcb0cb2', {}]],
  // H6's retained model loader reads typed-array constructors. These exact
  // cold helpers do not receive the renderer or mutate pass prototypes; keep
  // their original native-access facts without the caller reflection guard.
  ['GLTFLoader.js', ['131c0f78c01d19368ae495caa65b3adaa10487810a36a05bb5901b769a35ac16', {}]],
  ['BufferGeometryUtils.js', ['9fb63427ce6641fa14fd0baff9cc4d1b5f9c3d85fd084bf2e90e803c44ec1797', {}]],
  ['SkeletonUtils.js', ['b1632a703206c3d830de9fcbe515696770d04b71a15ee6b50afa6d2c3298c86f', {}]],
]);
const byHash = new Map([...pins].map(([name, [hash, exports]]) => ['sha256:' + hash, {name, exports}]));
const transformed = new Set(['MaskPass.js', 'EffectComposer.js', 'BloomPass.js']);

export function verifiedPostProcessingSources(modules, bases) {
  const result = new Map(), visiting = new Set();
  const verify = id => {
    if (result.has(id)) return true;
    if (visiting.has(id)) return false;
    const mod = modules.get(id), pin = byHash.get(mod?.content_hash);
    if (!pin || mod.dynamic_imports?.length) return false;
    visiting.add(id);
    const valid = mod.static_imports.every(request => request.specifier === 'three' ? bases.has(request.resolved_id) :
      modules.get(request.resolved_id)?.content_hash === 'sha256:' + pins.get(request.specifier.split('/').at(-1))?.[0] && verify(request.resolved_id));
    visiting.delete(id);
    if (valid) result.set(id, {...pin, sourceHash: mod.content_hash, transform: transformed.has(pin.name)});
    return valid;
  };
  for (const id of modules.keys()) verify(id);
  return result;
}

/** Keep the original classes, exports and dependency identities. Only these
 * pinned native-state expressions use compiler-private portable operations. */
export function transformThreePostProcessing(code, record,
  runtime = new URL('./three_pass_state.mjs', import.meta.url).href) {
  if (!record?.transform) return null;
  if ('sha256:' + createHash('sha256').update(code).digest('hex') !== record.sourceHash ||
      byHash.get(record.sourceHash)?.name !== record.name)
    throw new Error('Pinned postprocessing source changed before stencil lowering');
  const state = '__f3dPassState', constants = '__f3dStencilConstants';
  // Exact fingerprints make these finite spellings unambiguous. No source
  // expression or getter is duplicated; the exact-renderer helpers preserve
  // its original state/context accesses.
  code = code.replace(/\b(this\.renderer|renderer)\.getContext\(\)/g, (_, receiver) => `${constants}(${receiver})`)
    .replace(/\b(this\.renderer|renderer)\.state\b/g, (_, receiver) => `${state}(${receiver})`);
  return `import {getThreePassState as ${state}, getThreeStencilConstants as ${constants}} from ${JSON.stringify(runtime)};\n` + code;
}
