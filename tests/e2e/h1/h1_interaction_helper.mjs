/**
 * @file tests/e2e/h1/h1_interaction_helper.mjs
 * H1 interaction and reference-equality navigation helper (Plan §3.4, §5.1, §6.7; Mails #7594, #7595, #7694).
 *
 * Provides standalone, reusable functions to:
 * 1. Build and parse H1 URLs for both unmodified upstream reference and routed facade endpoints,
 *    strictly preserving count=0 and all query parameters.
 * 2. Locate and drive genuine Three.js Inspector controls (renderBundle, webgpu, dynamic) via standard DOM events.
 * 3. Extract comprehensive observable snapshots from active H1 window/document instances across iframe realms.
 * 4. Compare reference vs candidate snapshots asserting mandatory canvas initialization, mandatory control
 *    presence, dynamic/renderBundle/webgpu state parity, and honest candidate route attribution without false passes.
 */

export const H1_PATHS = Object.freeze({
  ROUTED: '/examples/webgpu_performance_renderbundle.html',
  REFERENCE: '/upstream/three.js/examples/webgpu_performance_renderbundle.html',
});

/**
 * Constructs a full or relative H1 URL with query parameters.
 * Preserves H1's query parameter keys and formatting, correctly preserving count=0.
 *
 * @param {string} [baseUrl=''] Optional base URL prefix (e.g. 'http://127.0.0.1:8080')
 * @param {Object} [options]
 * @param {boolean} [options.isReference=false] Whether to target unmodified upstream reference
 * @param {'webgpu'|'webgl'} [options.backend='webgpu'] Backend selection
 * @param {boolean} [options.renderBundle=true] Whether renderBundle is enabled
 * @param {number} [options.count=4000] Object count query parameter (preserves 0)
 * @returns {string} Fully constructed URL
 */
export function buildH1Url(baseUrl = '', {
  isReference = false,
  backend = 'webgpu',
  renderBundle = true,
  count = 4000,
} = {}) {
  const base = (baseUrl || '').replace(/\/+$/, '');
  const pathname = isReference ? H1_PATHS.REFERENCE : H1_PATHS.ROUTED;
  const backendParam = `backend=${backend === 'webgl' ? 'webgl' : 'webgpu'}`;
  const renderBundleParam = `&renderBundle=${renderBundle ? 'true' : 'false'}`;
  const parsedCount = count !== undefined && count !== null && Number.isFinite(Number(count))
    ? Number(count)
    : 4000;
  const countParam = `&count=${parsedCount}`;

  return `${base}${pathname}?${backendParam}${renderBundleParam}${countParam}`;
}

/**
 * Parses query parameters according to H1's exact startup semantics:
 * - api.webgpu = searchParams.get('backend') !== 'webgl'
 * - api.renderBundle = searchParams.get('renderBundle') !== 'false'
 * - api.count = parseFloat(searchParams.get('count') || 4000) (preserves 0)
 *
 * @param {string} searchStringOrUrl Query string (e.g. '?backend=webgl') or full URL
 * @returns {{ backend: 'webgpu'|'webgl', webgpu: boolean, renderBundle: boolean, count: number }}
 */
export function parseH1QueryParams(searchStringOrUrl = '') {
  let search = searchStringOrUrl || '';
  const queryIndex = search.indexOf('?');
  if (queryIndex !== -1) {
    search = search.slice(queryIndex);
  }

  const params = new URLSearchParams(search);
  const isWebGPU = params.get('backend') !== 'webgl';
  const renderBundle = params.get('renderBundle') !== 'false';
  const count = parseFloat(params.get('count') || 4000);

  return {
    backend: isWebGPU ? 'webgpu' : 'webgl',
    webgpu: isWebGPU,
    renderBundle,
    count,
  };
}

/**
 * Computes the next relative redirect URL produced by H1's original reload() function:
 * location.href = location.pathname + '?' + backendParam + renderBundleParam + countParam;
 * Preserves count=0.
 *
 * @param {string} currentPath Current location.pathname
 * @param {Object} nextState Target state
 * @param {'webgpu'|'webgl'} [nextState.backend='webgpu']
 * @param {boolean} [nextState.renderBundle=true]
 * @param {number} [nextState.count=4000]
 * @returns {string} Next URL path + query string
 */
export function computeH1ReloadUrl(currentPath, {
  backend = 'webgpu',
  renderBundle = true,
  count = 4000,
} = {}) {
  const backendParam = `backend=${backend === 'webgl' ? 'webgl' : 'webgpu'}`;
  const renderBundleParam = `&renderBundle=${renderBundle ? 'true' : 'false'}`;
  const parsedCount = count !== undefined && count !== null && Number.isFinite(Number(count))
    ? Number(count)
    : 4000;
  const countParam = `&count=${parsedCount}`;

  return `${currentPath}?${backendParam}${renderBundleParam}${countParam}`;
}

/**
 * Locates an Inspector parameter row and its input checkbox by label text.
 * Inspector renders parameters into `.list-item-row` elements where one cell contains
 * the property name span and another contains `label.custom-checkbox > input[type="checkbox"]`.
 *
 * @param {Document} doc Document to search
 * @param {string} label Label text to match (case-insensitive substring, e.g. 'render bundle', 'webgpu', 'dynamic')
 * @returns {{ found: boolean, row: Element|null, checkbox: HTMLInputElement|null, labelText: string|null }}
 */
export function findInspectorControl(doc, label) {
  if (!doc || typeof doc.querySelectorAll !== 'function') {
    return { found: false, row: null, checkbox: null, labelText: null };
  }

  const normalizedTarget = (label || '').trim().toLowerCase();
  const rows = doc.querySelectorAll('.list-item-row');

  for (const row of rows) {
    const text = (row.textContent || '').trim().toLowerCase();
    if (text.includes(normalizedTarget)) {
      const checkbox = row.querySelector('input[type="checkbox"]');
      return {
        found: true,
        row,
        checkbox,
        labelText: (row.textContent || '').trim(),
      };
    }
  }

  // Fallback: search across all input checkboxes if row structure differs
  const checkboxes = doc.querySelectorAll('input[type="checkbox"]');
  for (const cb of checkboxes) {
    const parentRow = cb.closest('.list-item-row') || cb.parentElement;
    const parentText = (parentRow?.textContent || '').trim().toLowerCase();
    if (parentText.includes(normalizedTarget)) {
      return {
        found: true,
        row: parentRow,
        checkbox: cb,
        labelText: (parentRow?.textContent || '').trim(),
      };
    }
  }

  return { found: false, row: null, checkbox: null, labelText: null };
}

/**
 * Toggles an Inspector checkbox control by simulating standard user interaction.
 * Updates `.checked` and dispatches a bubbling `change` event to invoke Inspector listeners.
 *
 * @param {Document} doc Document containing the Inspector
 * @param {string} label Control label (e.g. 'render bundle', 'webgpu', 'dynamic')
 * @returns {{ success: boolean, label: string, previousValue: boolean, nextValue: boolean, error?: string }}
 */
export function toggleInspectorControl(doc, label) {
  const control = findInspectorControl(doc, label);
  if (!control.found || !control.checkbox) {
    return {
      success: false,
      label,
      previousValue: false,
      nextValue: false,
      error: `Inspector control matching "${label}" not found or lacks checkbox input`,
    };
  }

  const cb = control.checkbox;
  const previousValue = Boolean(cb.checked);
  const nextValue = !previousValue;

  cb.checked = nextValue;

  // Dispatch bubbling change event
  let changeEvent;
  if (typeof Event === 'function') {
    changeEvent = new Event('change', { bubbles: true, cancelable: true });
  } else if (doc.createEvent) {
    changeEvent = doc.createEvent('Event');
    changeEvent.initEvent('change', true, true);
  }

  if (changeEvent) {
    cb.dispatchEvent(changeEvent);
  }

  return {
    success: true,
    label,
    previousValue,
    nextValue,
  };
}

/**
 * Identifies the WebGL or WebGPU context created on a canvas element across iframe realms.
 * Avoids single-realm instanceof checks that fail when canvas belongs to a separate Window realm.
 *
 * @param {HTMLCanvasElement} canvas Canvas element
 * @param {Window} [targetWindow=null] Target window owning the canvas
 * @returns {'webgpu'|'webgl2'|'webgl'|'2d'|'bitmaprenderer'|'none'}
 */
export function detectCanvasContext(canvas, targetWindow = null) {
  if (!canvas || typeof canvas.getContext !== 'function') return 'none';

  const win = targetWindow || canvas.ownerDocument?.defaultView || (typeof window !== 'undefined' ? window : null);

  // 1. Check WebGPU context
  try {
    const gpuCtx = canvas.getContext('webgpu');
    if (gpuCtx) {
      const isWebGPU = (win?.GPUCanvasContext && gpuCtx instanceof win.GPUCanvasContext) ||
        gpuCtx.constructor?.name === 'GPUCanvasContext' ||
        Object.prototype.toString.call(gpuCtx) === '[object GPUCanvasContext]';
      if (isWebGPU) return 'webgpu';
    }
  } catch (_) {}

  // 2. Check WebGL2 context
  try {
    const gl2 = canvas.getContext('webgl2');
    if (gl2) {
      const isWebGL2 = (win?.WebGL2RenderingContext && gl2 instanceof win.WebGL2RenderingContext) ||
        gl2.constructor?.name === 'WebGL2RenderingContext' ||
        Object.prototype.toString.call(gl2) === '[object WebGL2RenderingContext]';
      if (isWebGL2) return 'webgl2';
    }
  } catch (_) {}

  // 3. Check WebGL1 context
  try {
    const gl1 = canvas.getContext('webgl');
    if (gl1) {
      const isWebGL = (win?.WebGLRenderingContext && gl1 instanceof win.WebGLRenderingContext) ||
        gl1.constructor?.name === 'WebGLRenderingContext' ||
        Object.prototype.toString.call(gl1) === '[object WebGLRenderingContext]';
      if (isWebGL) return 'webgl';
    }
  } catch (_) {}

  return 'none';
}

/**
 * Extracts Inspector log messages from the document DOM.
 *
 * @param {Document} doc Document to search
 * @returns {{ sign: string, all: string[] }}
 */
export function extractInspectorLogs(doc) {
  if (!doc || typeof doc.querySelectorAll !== 'function') {
    return { sign: '', all: [] };
  }

  const logElements = doc.querySelectorAll(
    '.three-inspector .console-log .log-message, .log-message, .inspector-log, .log-item, .item-log'
  );
  const all = Array.from(logElements).map(el => (el.textContent || '').trim()).filter(Boolean);
  const sign = all.find(txt => txt.includes('THREE.WebGPURenderer')) || all[0] || '';

  return { sign, all };
}

/**
 * Captures a complete observable snapshot from an active H1 window and document across iframe realms.
 * Works uniformly on both unmodified upstream reference and routed candidate windows.
 *
 * @param {Window} win Target window
 * @param {Document} doc Target document
 * @returns {Object} Observable state snapshot
 */
export function readH1Snapshot(win, doc) {
  const currentUrl = win.location?.href || '';
  const pathname = win.location?.pathname || '';
  const search = win.location?.search || '';
  const query = parseH1QueryParams(search);

  const canvas = doc.querySelector ? doc.querySelector('canvas') : null;
  const contextType = detectCanvasContext(canvas, win);

  const renderBundleCtrl = findInspectorControl(doc, 'render bundle');
  const webgpuCtrl = findInspectorControl(doc, 'webgpu');
  const dynamicCtrl = findInspectorControl(doc, 'dynamic');

  const logs = extractInspectorLogs(doc);
  const router = win.__f3d_router__ || null;
  const isRouted = Boolean(router);
  const decisionLog = router && typeof router.getDecisionLog === 'function'
    ? router.getDecisionLog()
    : [];

  const activeDecision = decisionLog.find(d => d.site === 'WebGPURenderer') || null;

  return {
    url: currentUrl,
    pathname,
    search,
    query,
    title: doc.title || '',
    canvas: {
      exists: Boolean(canvas),
      contextType,
      width: canvas ? (canvas.width || 0) : 0,
      height: canvas ? (canvas.height || 0) : 0,
    },
    controls: {
      renderBundle: {
        found: renderBundleCtrl.found,
        checked: Boolean(renderBundleCtrl.checkbox?.checked),
      },
      webgpu: {
        found: webgpuCtrl.found,
        checked: Boolean(webgpuCtrl.checkbox?.checked),
      },
      dynamic: {
        found: dynamicCtrl.found,
        checked: Boolean(dynamicCtrl.checkbox?.checked),
      },
    },
    inspector: {
      sign: logs.sign,
      logs: logs.all,
    },
    routing: {
      isRouted,
      activeRoute: activeDecision ? activeDecision.route : null,
      activeReasons: activeDecision ? (activeDecision.reasons || []) : [],
      decisionCount: decisionLog.length,
    },
  };
}

/**
 * Compares an unmodified upstream reference snapshot against a candidate routed snapshot.
 * Rejects false passes:
 * 1. Requires real canvas initialized with valid GPU/GL context on BOTH reference and candidate.
 * 2. Requires all 3 Inspector controls (renderBundle, webgpu, dynamic) present on BOTH.
 * 3. Enforces dynamic, renderBundle, and webgpu state parity.
 * 4. Asserts exact query parameter parity (preserving count=0).
 * 5. Validates candidate honest route attribution and reference router purity.
 *
 * @param {Object} reference Upstream reference snapshot
 * @param {Object} candidate Routed candidate snapshot
 * @returns {{ pass: boolean, diffs: string[], details: Object }}
 */
export function compareH1Snapshots(reference, candidate) {
  const diffs = [];

  if (!reference || !candidate) {
    return {
      pass: false,
      diffs: ['Reference or candidate snapshot is null or undefined'],
      details: { reference, candidate },
    };
  }

  // 0. Query object presence
  if (!reference.query || typeof reference.query !== 'object') {
    diffs.push('Reference snapshot missing valid query object');
  }
  if (!candidate.query || typeof candidate.query !== 'object') {
    diffs.push('Candidate snapshot missing valid query object');
  }

  // 1. Mandatory Canvas Presence & Valid Context Enum (webgpu, webgl2, webgl)
  const VALID_CONTEXTS = ['webgpu', 'webgl2', 'webgl'];
  if (!reference.canvas?.exists || !VALID_CONTEXTS.includes(reference.canvas.contextType)) {
    diffs.push(`Reference canvas must be initialized with valid GPU/GL context (${VALID_CONTEXTS.join('/')}, got contextType '${reference.canvas?.contextType || 'none'}', exists: ${Boolean(reference.canvas?.exists)})`);
  }
  if (!candidate.canvas?.exists || !VALID_CONTEXTS.includes(candidate.canvas.contextType)) {
    diffs.push(`Candidate canvas must be initialized with valid GPU/GL context (${VALID_CONTEXTS.join('/')}, got contextType '${candidate.canvas?.contextType || 'none'}', exists: ${Boolean(candidate.canvas?.exists)})`);
  }

  // Canvas context type parity
  if (reference.canvas?.contextType !== candidate.canvas?.contextType) {
    diffs.push(`Canvas context mismatch: reference '${reference.canvas?.contextType}' vs candidate '${candidate.canvas?.contextType}'`);
  }

  // Context enum must match requested branch (webgpu vs webgl2 for this pinned H1)
  if (candidate.query?.backend) {
    const expectedContext = candidate.query.backend === 'webgl' ? 'webgl2' : 'webgpu';
    if (candidate.canvas?.contextType && candidate.canvas.contextType !== expectedContext) {
      diffs.push(`Canvas context '${candidate.canvas.contextType}' does not match requested branch '${expectedContext}' for backend '${candidate.query.backend}'`);
    }
  }

  // 2. Query parameter parity (using Object.is to strictly preserve count comparisons including NaN)
  if (reference.query && candidate.query) {
    if (reference.query.backend !== candidate.query.backend) {
      diffs.push(`Backend query mismatch: reference '${reference.query.backend}' vs candidate '${candidate.query.backend}'`);
    }
    if (reference.query.renderBundle !== candidate.query.renderBundle) {
      diffs.push(`renderBundle query mismatch: reference ${reference.query.renderBundle} vs candidate ${candidate.query.renderBundle}`);
    }
    if (!Object.is(reference.query.count, candidate.query.count)) {
      diffs.push(`count query mismatch: reference ${reference.query.count} vs candidate ${candidate.query.count}`);
    }
  }

  // 3. Document title parity
  if (reference.title !== candidate.title) {
    diffs.push(`Document title mismatch: reference '${reference.title}' vs candidate '${candidate.title}'`);
  }

  // 4. Mandatory presence & state parity of Inspector controls (renderBundle, webgpu, dynamic)
  // 4a. renderBundle control
  if (!reference.controls?.renderBundle?.found) {
    diffs.push('renderBundle control missing from reference Inspector');
  }
  if (!candidate.controls?.renderBundle?.found) {
    diffs.push('renderBundle control missing from candidate Inspector');
  }
  if (reference.controls?.renderBundle?.found && candidate.controls?.renderBundle?.found) {
    if (Boolean(reference.controls.renderBundle.checked) !== Boolean(candidate.controls.renderBundle.checked)) {
      diffs.push(`renderBundle checked state mismatch: reference ${reference.controls.renderBundle.checked} vs candidate ${candidate.controls.renderBundle.checked}`);
    }
  }

  // 4b. webgpu control
  if (!reference.controls?.webgpu?.found) {
    diffs.push('webgpu control missing from reference Inspector');
  }
  if (!candidate.controls?.webgpu?.found) {
    diffs.push('webgpu control missing from candidate Inspector');
  }
  if (reference.controls?.webgpu?.found && candidate.controls?.webgpu?.found) {
    if (Boolean(reference.controls.webgpu.checked) !== Boolean(candidate.controls.webgpu.checked)) {
      diffs.push(`webgpu checked state mismatch: reference ${reference.controls.webgpu.checked} vs candidate ${candidate.controls.webgpu.checked}`);
    }
  }

  // 4c. dynamic control
  if (!reference.controls?.dynamic?.found) {
    diffs.push('dynamic control missing from reference Inspector');
  }
  if (!candidate.controls?.dynamic?.found) {
    diffs.push('dynamic control missing from candidate Inspector');
  }
  if (reference.controls?.dynamic?.found && candidate.controls?.dynamic?.found) {
    if (Boolean(reference.controls.dynamic.checked) !== Boolean(candidate.controls.dynamic.checked)) {
      diffs.push(`dynamic checked state mismatch: reference ${reference.controls.dynamic.checked} vs candidate ${candidate.controls.dynamic.checked}`);
    }
  }

  // 5. Candidate honest routing attribution
  if (!candidate.routing?.isRouted) {
    diffs.push('Candidate must be served through routed compatibility facade (__f3d_router__ present)');
  } else {
    const expectedRoute = candidate.query?.backend === 'webgl' ? 'exact-backend' : 'retained-upstream';
    const expectedReason = candidate.query?.backend === 'webgl' ? 'explicit-source-selection' : 'specialization-unavailable';

    if (candidate.routing.activeRoute !== expectedRoute) {
      diffs.push(`Candidate route dishonest: expected '${expectedRoute}', got '${candidate.routing.activeRoute}'`);
    }
    if (!Array.isArray(candidate.routing.activeReasons) || !candidate.routing.activeReasons.includes(expectedReason)) {
      diffs.push(`Candidate reason missing: expected '${expectedReason}', got [${(candidate.routing.activeReasons || []).join(', ')}]`);
    }
  }

  // 6. Reference purity assertion: reference must NOT have router
  if (reference.routing?.isRouted) {
    diffs.push('Reference snapshot unexpectedly contains __f3d_router__ (must be unmodified upstream)');
  }

  return {
    pass: diffs.length === 0,
    diffs,
    details: {
      backend: candidate.query?.backend,
      renderBundle: candidate.query?.renderBundle,
      count: candidate.query?.count,
      canvasContext: candidate.canvas?.contextType,
      candidateRoute: candidate.routing?.activeRoute,
      diffCount: diffs.length,
    },
  };
}

/**
 * Compares two pixel buffers asserting exact dimensions, calculating per-channel delta,
 * diff pixel percentage, RMSE, and enforcing Plan §6 (line 1227) 0.1% different-pixel limit.
 *
 * @param {Object} refPixels Reference pixel buffer { width, height, data }
 * @param {Object} candPixels Candidate pixel buffer { width, height, data }
 * @param {Object} [options]
 * @param {number} [options.colorTolerance=2] Maximum allowed per-channel delta before pixel is counted as different
 * @param {number} [options.maxDiffPixelPercent=0.1] Maximum allowed percentage of different pixels (Plan §6 / line 1227)
 * @returns {{
 *   pass: boolean,
 *   totalPixels: number,
 *   diffPixels: number,
 *   diffPercent: number,
 *   maxChannelDiff: number,
 *   rmse: number,
 *   colorTolerance: number,
 *   maxDiffPixelPercent: number,
 *   error?: string
 * }}
 */
export function comparePixelBuffers(refPixels, candPixels, {
  colorTolerance = 2,
  maxDiffPixelPercent = 0.1,
} = {}) {
  if (!refPixels || !candPixels || !refPixels.data || !candPixels.data) {
    return {
      pass: false,
      passed: false,
      totalPixels: 0,
      diffPixels: 0,
      diffPercent: 100,
      maxChannelDiff: 255,
      rmse: 255,
      colorTolerance,
      maxDiffPixelPercent,
      error: 'Missing or invalid pixel buffer data',
    };
  }

  if (refPixels.width !== candPixels.width || refPixels.height !== candPixels.height) {
    return {
      pass: false,
      passed: false,
      totalPixels: 0,
      diffPixels: 0,
      diffPercent: 100,
      maxChannelDiff: 255,
      rmse: 255,
      colorTolerance,
      maxDiffPixelPercent,
      error: `Dimension mismatch: reference ${refPixels.width}x${refPixels.height} vs candidate ${candPixels.width}x${candPixels.height}`,
    };
  }

  const totalPixels = refPixels.width * refPixels.height;
  const len = totalPixels * 4;
  if (refPixels.data.length < len || candPixels.data.length < len) {
    return {
      pass: false,
      passed: false,
      totalPixels,
      diffPixels: totalPixels,
      diffPercent: 100,
      maxChannelDiff: 255,
      rmse: 255,
      colorTolerance,
      maxDiffPixelPercent,
      error: `Pixel byte array length insufficient for ${refPixels.width}x${refPixels.height}`,
    };
  }

  let diffPixels = 0;
  let maxChannelDiff = 0;
  let sumSquaredDiff = 0;

  for (let i = 0; i < len; i += 4) {
    const dr = Math.abs(refPixels.data[i] - candPixels.data[i]);
    const dg = Math.abs(refPixels.data[i + 1] - candPixels.data[i + 1]);
    const db = Math.abs(refPixels.data[i + 2] - candPixels.data[i + 2]);
    const da = Math.abs(refPixels.data[i + 3] - candPixels.data[i + 3]);

    const pixelDiff = Math.max(dr, dg, db, da);
    if (pixelDiff > maxChannelDiff) maxChannelDiff = pixelDiff;
    sumSquaredDiff += (dr * dr + dg * dg + db * db) / 3;

    if (dr > colorTolerance || dg > colorTolerance || db > colorTolerance || da > colorTolerance) {
      diffPixels++;
    }
  }

  const diffPercent = (diffPixels / totalPixels) * 100;
  const rmse = Math.sqrt(sumSquaredDiff / totalPixels);
  const pass = diffPercent <= maxDiffPixelPercent;

  return {
    pass,
    passed: pass,
    totalPixels,
    diffPixels,
    diffPercent,
    maxChannelDiff,
    rmse,
    colorTolerance,
    maxDiffPixelPercent,
  };
}

/**
 * Creates a deliberately mutated copy of a pixel buffer for strict negative testing (Mail #13025).
 * Mutates a block of pixels exceeding the 0.1% threshold to verify rejection.
 *
 * @param {Object} pixelBuffer { width, height, data }
 * @param {number} [mutatePercent=1.0] Percentage of pixels to mutate (must exceed 0.1%)
 * @returns {{ width: number, height: number, data: Uint8ClampedArray }}
 */
export function createMismatchedPixelBuffer(pixelBuffer, mutatePercent = 1.0) {
  const width = pixelBuffer.width || 400;
  const height = pixelBuffer.height || 300;
  const copy = new Uint8ClampedArray(pixelBuffer.data);
  const totalPixels = width * height;
  const pixelsToMutate = Math.max(10, Math.ceil((totalPixels * mutatePercent) / 100));

  for (let p = 0; p < pixelsToMutate; p++) {
    const idx = p * 4;
    if (idx + 3 < copy.length) {
      // Invert color channels and set full alpha to guarantee mismatch
      copy[idx] = 255 - copy[idx];
      copy[idx + 1] = 255 - copy[idx + 1];
      copy[idx + 2] = 255 - copy[idx + 2];
      copy[idx + 3] = 255;
    }
  }

  return {
    width,
    height,
    data: copy,
  };
}

/**
 * Captures pixel buffer from an active HTMLCanvasElement at actual native dimensions.
 * Avoids any downscaling or image resampling to pass thresholds (ROOT H1 REVIEW).
 *
 * @param {HTMLCanvasElement} canvas Source canvas
 * @param {number} [targetWidth] Optional explicit width override (defaults to native canvas.width)
 * @param {number} [targetHeight] Optional explicit height override (defaults to native canvas.height)
 * @returns {{ width: number, height: number, data: Uint8ClampedArray }}
 */
export function captureCanvasPixels(canvas, targetWidth, targetHeight) {
  if (!canvas) throw new Error('Canvas element is required for pixel capture');

  const doc = canvas.ownerDocument || (typeof document !== 'undefined' ? document : null);
  if (!doc || typeof doc.createElement !== 'function') {
    throw new Error('Document environment required for canvas pixel capture');
  }

  // Use actual native canvas dimensions by default; no downscaling or resampling (ROOT H1 REVIEW)
  const width = targetWidth || canvas.width || 800;
  const height = targetHeight || canvas.height || 600;

  const offscreen = doc.createElement('canvas');
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Failed to acquire 2D rendering context for canvas readback');

  // Direct 1:1 pixel copy at native resolution
  ctx.drawImage(canvas, 0, 0, width, height, 0, 0, width, height);

  const imgData = ctx.getImageData(0, 0, width, height);
  return {
    width,
    height,
    data: imgData.data,
  };
}


/**
 * Deterministic pseudo-random sequence prelude for matched H1 testing (ROOT H1 REVIEW).
 * Overrides Math.random before application scripts run to ensure identical geometry,
 * transforms, and material colors across reference and candidate runs.
 * Exposes seed reset callback and read-only invocation counter.
 */
export const SEEDED_RANDOM_PRELUDE = `
<script id="f3d-seeded-random-prelude">
// Deterministic pseudo-random sequence for matched H1 testing (ROOT H1 REVIEW)
(function() {
  const INITIAL_SEED = 0x12345678;
  let s = INITIAL_SEED;
  let count = 0;
  function seededRandom() {
    count++;
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  Math.random = seededRandom;
  if (typeof window !== 'undefined') {
    Object.defineProperty(window, '__f3d_rng_count__', {
      get: () => count,
      configurable: true,
    });
    Object.defineProperty(window, '__f3d_rng_state__', {
      get: () => s,
      configurable: true,
    });
  }
})();
</script>
`;

/**
 * Injects the seeded Math.random prelude before application scripts in HTML.
 *
 * @param {string} html Target HTML source
 * @returns {string} Injected HTML
 */
export function injectSeededRandomPrelude(html) {
  if (typeof html !== 'string') return html;
  if (html.includes('id="f3d-seeded-random-prelude"')) return html;

  const importMapIndex = html.indexOf('<script type="importmap">');
  if (importMapIndex !== -1) {
    const endTag = '</script>';
    const closeIndex = html.indexOf(endTag, importMapIndex);
    if (closeIndex !== -1) {
      const insertPos = closeIndex + endTag.length;
      return html.slice(0, insertPos) + '\n\t\t' + SEEDED_RANDOM_PRELUDE + html.slice(insertPos);
    }
  }
  if (html.includes('<script type="module">')) {
    return html.replace('<script type="module">', `${SEEDED_RANDOM_PRELUDE}\n\t\t<script type="module">`);
  }
  if (html.includes('<script')) {
    return html.replace('<script', `${SEEDED_RANDOM_PRELUDE}\n\t\t<script`);
  }
  if (html.includes('</head>')) {
    return html.replace('</head>', `${SEEDED_RANDOM_PRELUDE}\n</head>`);
  }
  return `${SEEDED_RANDOM_PRELUDE}\n${html}`;
}

/**
 * Extracts read-only checkpoint observations (first mesh transform, color, camera/projection, RNG count).
 * NEVER copies or mutates state (ROOT H1 REVIEW).
 *
 * @param {Window} win
 * @param {Document} doc
 * @returns {Object|null}
 */
export function extractCheckpointObservations(win, doc) {
  if (!win) return null;
  const scene = win.__f3d_last_scene__;
  const camera = win.__f3d_last_camera__;
  const group = scene && scene.children ? scene.children.find(c => c.isGroup || c.isBundleGroup) : null;
  const firstMesh = group && group.children ? group.children[0] : null;

  let firstMeshObs = null;
  if (firstMesh) {
    firstMeshObs = {
      position: firstMesh.position ? [firstMesh.position.x, firstMesh.position.y, firstMesh.position.z] : null,
      quaternion: firstMesh.quaternion ? [firstMesh.quaternion.x, firstMesh.quaternion.y, firstMesh.quaternion.z, firstMesh.quaternion.w] : null,
      scale: firstMesh.scale ? [firstMesh.scale.x, firstMesh.scale.y, firstMesh.scale.z] : null,
      colorHex: firstMesh.material && firstMesh.material.color && typeof firstMesh.material.color.getHexString === 'function'
        ? firstMesh.material.color.getHexString()
        : null,
      matrix: firstMesh.matrix && firstMesh.matrix.elements ? Array.from(firstMesh.matrix.elements) : null,
    };
  }

  let cameraObs = null;
  if (camera) {
    cameraObs = {
      position: camera.position ? [camera.position.x, camera.position.y, camera.position.z] : null,
      quaternion: camera.quaternion ? [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w] : null,
      projectionMatrix: camera.projectionMatrix && camera.projectionMatrix.elements ? Array.from(camera.projectionMatrix.elements) : null,
      aspect: camera.aspect !== undefined ? camera.aspect : null,
      fov: camera.fov !== undefined ? camera.fov : null,
      near: camera.near !== undefined ? camera.near : null,
      far: camera.far !== undefined ? camera.far : null,
    };
  }

  const rngCount = typeof win.__f3d_rng_count__ === 'number' ? win.__f3d_rng_count__ : null;
  const rngState = typeof win.__f3d_rng_state__ === 'number' ? win.__f3d_rng_state__ : null;

  return {
    firstMesh: firstMeshObs,
    camera: cameraObs,
    rng: {
      count: rngCount,
      state: rngState,
    },
  };
}

/**
 * Compares read-only checkpoint observations between reference and candidate without copying (ROOT H1 REVIEW).
 *
 * @param {Object} refObs Reference observations
 * @param {Object} candObs Candidate observations
 * @returns {{ pass: boolean, diffs: string[] }}
 */
export function compareCheckpointObservations(refObs, candObs) {
  const diffs = [];
  if (!refObs || !candObs) {
    diffs.push('Missing checkpoint observations (ref or cand null)');
    return { pass: false, diffs };
  }

  // 1. RNG invocation count & state agreement
  if (refObs.rng && candObs.rng) {
    if (refObs.rng.count !== candObs.rng.count) {
      diffs.push(`RNG invocation count mismatch: ref=${refObs.rng.count} vs cand=${candObs.rng.count}`);
    }
    if (refObs.rng.state !== candObs.rng.state) {
      diffs.push(`RNG state mismatch: ref=${refObs.rng.state} vs cand=${candObs.rng.state}`);
    }
  }

  // 2. First mesh transform and material color
  if (refObs.firstMesh && candObs.firstMesh) {
    if (refObs.firstMesh.colorHex !== candObs.firstMesh.colorHex) {
      diffs.push(`First mesh color mismatch: ref=${refObs.firstMesh.colorHex} vs cand=${candObs.firstMesh.colorHex}`);
    }
    if (refObs.firstMesh.position && candObs.firstMesh.position) {
      for (let i = 0; i < 3; i++) {
        if (Math.abs(refObs.firstMesh.position[i] - candObs.firstMesh.position[i]) > 1e-4) {
          diffs.push(`First mesh position[${i}] mismatch: ref=${refObs.firstMesh.position[i].toFixed(4)} vs cand=${candObs.firstMesh.position[i].toFixed(4)}`);
        }
      }
    }
    if (refObs.firstMesh.scale && candObs.firstMesh.scale) {
      for (let i = 0; i < 3; i++) {
        if (Math.abs(refObs.firstMesh.scale[i] - candObs.firstMesh.scale[i]) > 1e-4) {
          diffs.push(`First mesh scale[${i}] mismatch: ref=${refObs.firstMesh.scale[i].toFixed(4)} vs cand=${candObs.firstMesh.scale[i].toFixed(4)}`);
        }
      }
    }
    if (refObs.firstMesh.quaternion && candObs.firstMesh.quaternion) {
      for (let i = 0; i < 4; i++) {
        if (Math.abs(refObs.firstMesh.quaternion[i] - candObs.firstMesh.quaternion[i]) > 1e-4) {
          diffs.push(`First mesh quaternion[${i}] mismatch: ref=${refObs.firstMesh.quaternion[i].toFixed(4)} vs cand=${candObs.firstMesh.quaternion[i].toFixed(4)}`);
        }
      }
    }
  } else {
    diffs.push('First mesh observation missing in reference or candidate');
  }

  // 3. Camera projection matrix and parameters
  if (refObs.camera && candObs.camera) {
    if (refObs.camera.aspect !== null && candObs.camera.aspect !== null) {
      if (Math.abs(refObs.camera.aspect - candObs.camera.aspect) > 1e-4) {
        diffs.push(`Camera aspect mismatch: ref=${refObs.camera.aspect} vs cand=${candObs.camera.aspect}`);
      }
    }
    if (refObs.camera.projectionMatrix && candObs.camera.projectionMatrix) {
      for (let i = 0; i < 16; i++) {
        if (Math.abs(refObs.camera.projectionMatrix[i] - candObs.camera.projectionMatrix[i]) > 1e-4) {
          diffs.push(`Camera projectionMatrix[${i}] mismatch: ref=${refObs.camera.projectionMatrix[i].toFixed(4)} vs cand=${candObs.camera.projectionMatrix[i].toFixed(4)}`);
          break;
        }
      }
    }
  } else {
    diffs.push('Camera observation missing in reference or candidate');
  }

  return {
    pass: diffs.length === 0,
    diffs,
  };
}
