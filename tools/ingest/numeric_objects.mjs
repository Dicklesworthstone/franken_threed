/** Allocation witnesses emitted only inside verified, non-derived constructors.
 * A Proxy around a registered object never acquires the underlying witness.
 * This is compiler-private ownership metadata, not a public Three.js facade.
 */
const objects = new WeakSet();
const add = WeakSet.prototype.add, has = WeakSet.prototype.has, apply = Reflect.apply;
export const registerNumericObject = object => { apply(add, objects, [object]); };
export const isNumericObject = object => apply(has, objects, [object]);
