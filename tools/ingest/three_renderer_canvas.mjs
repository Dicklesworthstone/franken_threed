/** Native canvas-session ownership for source CanvasTarget selection.
 *
 * The public target is independent of the canvas currently being submitted.
 * Accepted requests lease their destination until FIFO submission finishes;
 * removing an Inspector preview cannot unconfigure a queued producer.
 * Accepted work leases its actual canvas; live rebinding of the source
 * backend's CanvasTarget context cache is outside this contract.
 */
export function createThreeRendererCanvases(THREE, renderer, canvas, fail) {
  // The WebGL-authored facade also uses these sessions, but its source module
  // does not export CanvasTarget. Keep its existing private canvas state; the
  // native WebGPU facade always uses the actual source CanvasTarget class.
  const initial = typeof THREE.CanvasTarget === 'function' ? new THREE.CanvasTarget(canvas)
    : Object.assign(new THREE.EventDispatcher(), {domElement: canvas, _width: canvas.width, _height: canvas.height,
      _pixelRatio: 1, _viewport: new THREE.Vector4(0, 0, canvas.width, canvas.height),
      _scissor: new THREE.Vector4(0, 0, canvas.width, canvas.height), _scissorTest: false,
      dispose() { this.dispatchEvent({type: 'dispose'}); }});
  renderer._canvasTarget = initial;
  // Source CanvasTarget is authoritative, including direct changes made by
  // Inspector's Viewer. No second copy of viewport or logical size can drift.
  for (const key of ['_width', '_height', '_pixelRatio', '_viewport', '_scissor', '_scissorTest'])
    Object.defineProperty(renderer, key, {configurable: true,
      get() { return this._canvasTarget[key]; }, set(value) { this._canvasTarget[key] = value; }});
  Object.defineProperty(renderer, '_viewportDepth', {configurable: true,
    get() { return [this._viewport.minDepth ?? 0, this._viewport.maxDepth ?? 1]; },
    set(value) { [this._viewport.minDepth, this._viewport.maxDepth] = value; }});

  // Different source targets may wrap the same canvas (Viewer recreates a
  // split target on resize). Its native context still has exactly one owner.
  const entries = new Map();
  let active = null;
  const fields = ['_session', '_dispatcher', '_hdr', '_srgbView'];
  const validate = (target, canvas = target?.domElement) => {
    if ((target !== initial && !(typeof THREE.CanvasTarget === 'function' && target instanceof THREE.CanvasTarget)) ||
        typeof canvas?.getContext !== 'function')
      fail('SOURCE', 'Expected a pinned source CanvasTarget with a canvas');
    return target;
  };
  const save = () => {
    if (active) for (const field of fields) active[field] = renderer[field];
  };
  function release(entry) {
    entry._session?.dispose();
    for (const [target, listener] of entry.targets) target.removeEventListener('dispose', listener);
    entry.targets.clear();
    entries.delete(entry.canvas);
    if (active === entry) {
      active = null; renderer._session = null; renderer._dispatcher = null;
    }
  }
  function forget(target) {
    save();
    // domElement is mutable source state. Retire every session leased through
    // this target, including an accepted destination it no longer points to.
    for (const entry of [...entries.values()]) if (entry.targets.has(target)) {
      entry.retired = true;
      if (entry.references === 0) release(entry);
    }
  }
  function get(target, canvas = target.domElement) {
    validate(target, canvas);
    let entry = entries.get(canvas);
    if (!entry) {
      entry = {canvas, targets: new Map(), references: 0, retired: false,
        _session: null, _dispatcher: null, _hdr: false, _srgbView: true};
      entries.set(entry.canvas, entry);
    }
    if (!entry.targets.has(target)) {
      const listener = () => forget(target);
      entry.targets.set(target, listener); target.addEventListener('dispose', listener);
    }
    return entry;
  }
  return {
    select(target) { renderer._canvasTarget = validate(target); },
    retain(target, canvas = target.domElement) {
      const entry = get(target, canvas); entry.references++;
      let released = false;
      return () => {
        if (released) return; released = true;
        entry.references--; save();
        if (entry.retired && entry.references === 0 && entries.get(entry.canvas) === entry) release(entry);
      };
    },
    activate(target, canvas = target.domElement) {
      const entry = get(target, canvas);
      if (entry !== active) {
        save(); active = entry;
        for (const field of fields) renderer[field] = entry[field];
      }
    },
    matches(target, width, height, canvas = target.domElement) {
      return active?.canvas === canvas && renderer._session !== null &&
        renderer._session.width === width && renderer._session.height === height &&
        canvas.width === width && canvas.height === height;
    },
    save,
    forget,
    resize(target, width, height) {
      save();
      const session = entries.get(target.domElement)?._session;
      // A source resize may occur while another request prepares. Keep source
      // size observable now; the serialized request boundary resizes storage.
      if (session && !renderer._preparing && !renderer._drain) session.resize(width, height);
      else { target.domElement.width = width; target.domElement.height = height; }
    },
    async whenIdle() {
      save();
      await Promise.all([...entries.values()].map(entry => entry._session?.whenIdle()));
    },
    dispose() {
      save();
      for (const entry of [...entries.values()]) release(entry);
    },
  };
}
