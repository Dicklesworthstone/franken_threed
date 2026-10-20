/** Allocation witnesses emitted only inside verified, non-derived constructors.
 * A Proxy around a registered object never acquires the underlying witness.
 * This is compiler-private ownership metadata, not a public Three.js facade.
 */
const objects = new WeakSet();
export const registerNumericObject = object => { objects.add(object); };
export const isNumericObject = object => objects.has(object);
