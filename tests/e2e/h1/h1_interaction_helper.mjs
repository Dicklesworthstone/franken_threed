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
