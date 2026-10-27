/** Source BundleGroup records and their accepted resource lifetimes.
 * Native GPU command caching remains in animation_render_bundles.mjs. This
 * owner retains source selection; static changes refresh policy, not selection.
 */
import {createThreeNodeBundleGeometry} from './three_node_bundle_geometry.mjs';

export function createThreeNodeBundles(three, {maxEntries = 16384, maxBytes = 64 * 1024 * 1024,
  fail = (code, message) => { throw new Error(`${code}: ${message}`); }} = {}) {
  let groups = new WeakMap(), disposed = false;
  const entries = new Set(), wrappers = new WeakMap();
  const geometry = createThreeNodeBundleGeometry(three, {maxBytes, fail});
  const contextKey = (target, mrt, depth) => `${target === null ? 'default' :
    [target.textures.length, target.texture.format, target.texture.type, target.samples,
      target.depthBuffer, target.stencilBuffer].join(':')}-${mrt?.id ?? 'default'}-${depth}`;
  // A frame and its persistent bundle can hold the same upload generation.
  // Releasing a frame must not recapture current source resource values.
  function resources(owner) {
    if (!owner) return owner;
    let wrapper = wrappers.get(owner);
    if (wrapper) return wrapper;
    const refs = new Map(), methods = new Map();
    const retain = token => { refs.set(token, (refs.get(token) ?? 0) + 1); return token; };
    const release = token => {
      const count = refs.get(token);
      if (count > 1) { refs.set(token, count - 1); return; }
      refs.delete(token); owner.release(token);
    };
    wrapper = new Proxy({}, {get(_target, key) {
      if (key === 'retain') return retain;
      if (key === 'release') return release;
      const value = owner[key];
      if (typeof value !== 'function') return value;
      if (!methods.has(key)) methods.set(key, key === 'captureBinding' || key === 'capture' || key === 'captureShared'
        ? (...args) => { const token = value.apply(owner, args); return token === null ? null : retain(token); } : value.bind(owner));
      return methods.get(key);
    }});
    wrappers.set(owner, wrapper); return wrapper;
  }
  function releaseDraw(draw) {
    for (const binding of draw.nodeTextures) draw.bundleTextureOwner.release(binding.lease);
    for (const binding of draw.nodeStorage) draw.bundleStorageOwner.release(binding.lease);
    for (const lease of draw.nodeVertexStorage ?? []) draw.bundleStorageOwner.release(lease);
    draw.bundleGeometry?.release();
  }
  return Object.freeze({resources,
    select(group, camera, target, mrt, depth = 0) {
      if (disposed) fail('DISPOSED', 'Source bundle owner is disposed');
      let cameras = groups.get(group);
      if (!cameras) { cameras = new WeakMap(); groups.set(group, cameras); }
      let contexts = cameras.get(camera);
      if (!contexts) { contexts = new Map(); cameras.set(camera, contexts); }
      const key = contextKey(target, mrt, depth);
      let entry = contexts.get(key);
      if (!entry) {
        if (entries.size >= maxEntries) fail('LIMIT', 'Source bundle records exceed their capacity');
        entry = {group, camera, key, version: null, draws: null, tokens: new WeakMap()};
        contexts.set(key, entry); entries.add(entry);
      }
      return {entry, group, warm: entry.draws !== null && entry.version === group.version,
        opaque: [], transparent: []};
    },
    token(entry, object, material, passId) {
      let materials = entry.tokens.get(object);
      if (!materials) { materials = new WeakMap(); entry.tokens.set(object, materials); }
      let passes = materials.get(material);
      if (!passes) { passes = new Map(); materials.set(material, passes); }
      if (!passes.has(passId)) passes.set(passId, {});
      return passes.get(passId);
    },
    captureGeometry(view, attributes, renderId) {
      // Generated streams install into private views only. Freeze all streams
      // at the same upload boundary, before ordinary NodeFrame updates.
      attributes?.install();
      return geometry.capture(view, renderId);
    },
    commit(selection, draws, version = selection.group.version) {
      const retained = [], undo = [];
      try {
        for (const draw of draws) {
          const lease = draw.bundleGeometry?.retain() ?? null;
          if (lease) undo.push(() => lease.release());
          for (const binding of draw.nodeTextures) {
            draw.bundleTextureOwner.retain(binding.lease);
            undo.push(() => draw.bundleTextureOwner.release(binding.lease));
          }
          for (const binding of draw.nodeStorage) {
            draw.bundleStorageOwner.retain(binding.lease);
            undo.push(() => draw.bundleStorageOwner.release(binding.lease));
          }
          for (const vertexLease of draw.nodeVertexStorage ?? []) {
            draw.bundleStorageOwner.retain(vertexLease);
            undo.push(() => draw.bundleStorageOwner.release(vertexLease));
          }
          retained.push(Object.freeze({...draw, bundleGeometry: lease}));
        }
      } catch (error) {
        for (const release of undo.reverse()) release();
        throw error;
      }
      for (const previous of selection.entry.draws ?? []) releaseDraw(previous);
      selection.entry.draws = Object.freeze(retained);
      selection.entry.version = version;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of entries) for (const draw of entry.draws ?? []) releaseDraw(draw);
      entries.clear(); groups = new WeakMap(); geometry.dispose();
    },
  });
}
