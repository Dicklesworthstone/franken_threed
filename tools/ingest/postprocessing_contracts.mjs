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
  ['UnrealBloomPass.js', ['ba8f2fcadfa6588384c9473498f974d81d120f02f0e63a0e59c265202a006b5a', {UnrealBloomPass: 'pass'}]],
  ['BokehPass.js', ['5b9b0bc5cee6e24b2417da95f40eb2d178943c2ffbacbc4c2993a89d0597bfb3', {BokehPass: 'pass'}]],
  ['AfterimagePass.js', ['8cc0c8724a674a649d438c932e0b5f287e7d02f9fc35f627b669180bf4d2cc01', {AfterimagePass: 'pass'}]],
  ['SSAARenderPass.js', ['cd2b8e83b32e806091117dfbeff1f8f4e78df0175a4a3acd865b9989df1c420b', {SSAARenderPass: 'pass'}]],
  ['TAARenderPass.js', ['463dbda5d1f0f401b3575f60eddd2a84234ff5be44d29a3dc41129c34aba0f47', {TAARenderPass: 'pass'}]],
  ['SSAOPass.js', ['2f9c02a53e09917d039c07c5bd7a110f1ce39bc5ab2affa316ea94f5198ec586', {SSAOPass: 'pass'}]],
  ['GTAOPass.js', ['a277f848d1035c751c54d9706e579aeb74689126ced25723e53d52ac3f26f445', {GTAOPass: 'pass'}]],
  ['OutlinePass.js', ['8eb486b1436432cf6f080c4a55b027f6d56b3ea66a9735c140a8be986107667b', {OutlinePass: 'pass'}]],
  ['CopyShader.js', ['a33057d5ac91c43304c186ac0e8816e62bb2ed471d3a00ff3018dfd5c0389718', {}]],
  ['ConvolutionShader.js', ['bad71ed02351acb701e62e21a9d11f044c2c689d924d714061edb756fbea8ace', {}]],
  ['FilmShader.js', ['744c619c70f949af5d3f0f9edb27573743ab9ad12869e7c306ffb3d0caff7bb5', {}]],
  ['DotScreenShader.js', ['959a2e0ea3d4a89d67754e417c56bd67abd632427d51fab19361003cd0ca7cef', {}]],
  ['OutputShader.js', ['353479f77a8d7e2629d49ccac9fc2f5dbfdda5442e0adf867b00377a2fcb0cb2', {}]],
  ['LuminosityHighPassShader.js', ['5044f780b6e6cf863947f64c36fe1587132f7fbe395ada863cd1e5f0388dcf1e', {}]],
  ['BokehShader.js', ['bd8525158e3574a803f2e2ce5d5e263349a80ee41bf789e71446b9caa256aa9a', {}]],
  ['AfterimageShader.js', ['fb842ae657379a6ee3b431bd864b4e4956d36df14daf502bf6ffecdbf14de8bb', {}]],
  ['SSAOShader.js', ['034dad98927ed8f2677a19ce56d4da1212ce5070ed2666157106025dc260bb55', {}]],
  ['GTAOShader.js', ['cdea7be60452f61105375d2c18e4da2e8ce7b39a89246b1bb8fbf79319d58cfd', {}]],
  ['PoissonDenoiseShader.js', ['61da074a9ad8b2139dd164c96b8b62c299f056ce06c216e1f707c5151d997602', {}]],
  ['SimplexNoise.js', ['4b1455617135e6b477de76244de07547d95570643379b72f4758e50161b06072', {}]],
  // H6's retained model loader reads typed-array constructors. These exact
  // cold helpers do not receive the renderer or mutate pass prototypes; keep
  // their original native-access facts without the caller reflection guard.
  ['GLTFLoader.js', ['131c0f78c01d19368ae495caa65b3adaa10487810a36a05bb5901b769a35ac16', {}]],
  ['BufferGeometryUtils.js', ['9fb63427ce6641fa14fd0baff9cc4d1b5f9c3d85fd084bf2e90e803c44ec1797', {}]],
  ['SkeletonUtils.js', ['b1632a703206c3d830de9fcbe515696770d04b71a15ee6b50afa6d2c3298c86f', {}]],
]);
const byHash = new Map([...pins].map(([name, [hash, exports]]) => ['sha256:' + hash, {name, exports}]));
const transformed = new Set(['MaskPass.js', 'EffectComposer.js', 'BloomPass.js', 'UnrealBloomPass.js',
  'SSAARenderPass.js', 'TAARenderPass.js', 'SSAOPass.js', 'GTAOPass.js', 'OutlinePass.js']);

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
  const state = '__f3dPassState', constants = '__f3dStencilConstants', renderScene = '__f3dRenderPassScene',
    resizeComposer = '__f3dComposerResize';
  // Exact fingerprints make these finite spellings unambiguous. No source
  // expression or getter is duplicated; the exact-renderer helpers preserve
  // its original state/context accesses.
  code = code.replace(/\b(this\.renderer|renderer)\.getContext\(\)/g, (_, receiver) => `${constants}(${receiver})`)
    .replace(/\b(this\.renderer|renderer)\.state\b/g, (_, receiver) => `${state}(${receiver})`);
  // These source passes restore camera jitter before asynchronous preparation
  // completes. Mark only their exact scene-draw operation for accepted camera
  // capture; other pass work and source renderer calls retain their identities.
  if (['SSAARenderPass.js', 'TAARenderPass.js'].includes(record.name))
    code = code.replaceAll('renderer.render( this.scene, this.camera );', `${renderScene}(renderer, this.scene, this.camera);`);
  // AO excludes points/lines only for its normal/depth draw, then restores the
  // public objects. Retain that exact source list while preparation is queued.
  if (['SSAOPass.js', 'GTAOPass.js'].includes(record.name))
    code = code.replaceAll('renderer.render( this.scene, this.camera );',
      `${renderScene}(renderer, this.scene, this.camera, this._visibilityCache);`);
  // Each Outline draw's cache contains exactly the currently hidden objects:
  // selected meshes for depth, then nonselected objects for the selected mask.
  if (record.name === 'OutlinePass.js')
    code = code.replaceAll('renderer.render( this.renderScene, this.renderCamera );',
      `${renderScene}(renderer, this.renderScene, this.renderCamera, this._visibilityCache);`);
  if (record.name === 'EffectComposer.js') {
    // Retain the exact size calculation and pass iteration. The private scope
    // preserves accepted target generations across the source's resize-dispose
    // events, and does not turn explicit disposal into deferred resize.
    code = code.replace('\tsetSize( width, height ) {\n',
      `\tsetSize( width, height ) {\n\t\treturn ${resizeComposer}(this, () => {\n`)
      .replace('\t\t\tthis.passes[ i ].setSize( effectiveWidth, effectiveHeight );\n\n\t\t}\n\n\t}',
        '\t\t\tthis.passes[ i ].setSize( effectiveWidth, effectiveHeight );\n\n\t\t}\n\t\t});\n\t}');
  }
  return `import {getThreePassState as ${state}, getThreeStencilConstants as ${constants}, renderThreePassScene as ${renderScene}, withThreeComposerResize as ${resizeComposer}} from ${JSON.stringify(runtime)};\n` + code;
}
