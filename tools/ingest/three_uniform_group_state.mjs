/** Uniform-block identity shared by live materials and private pass snapshots.
 * A source dispose starts a new residency; already accepted captures retain
 * the former generation. The listener holds no renderer or snapshot owner.
 */
const generations = new WeakMap(), captures = new WeakMap();
function disposed({target}) { generations.set(target, {}); }
function generation(group) {
  let current = generations.get(group);
  if (!current) {
    current = {}; generations.set(group, current);
    group.addEventListener('dispose', disposed);
  }
  return current;
}

export function captureThreeUniformGroup(source, captured, uniforms) {
  captures.set(captured, {source, generation: generation(source), uniforms});
}

export function threeUniformGroupState(group) {
  return captures.get(group) ?? {source: group, generation: generation(group), uniforms: null};
}
