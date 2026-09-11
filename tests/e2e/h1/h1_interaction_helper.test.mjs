/**
 * @file tests/e2e/h1/h1_interaction_helper.test.mjs
 * Unit tests for H1 interaction helper and state comparison logic (Plan §3.4, §5.1, §6.7; Mails #7594, #7595, #7694).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  H1_PATHS,
  buildH1Url,
  parseH1QueryParams,
  computeH1ReloadUrl,
  findInspectorControl,
  toggleInspectorControl,
  detectCanvasContext,
  readH1Snapshot,
  compareH1Snapshots,
  comparePixelBuffers,
  createMismatchedPixelBuffer,
  captureCanvasPixels,
  injectSeededRandomPrelude,
  extractCheckpointObservations,
  compareCheckpointObservations,
} from './h1_interaction_helper.mjs';

test('H1 URL Builder: Generates exact query URLs for reference and routed variants, preserving count=0', () => {
  const base = 'http://127.0.0.1:8080';

  // Default routed WebGPU
  const urlRoutedDefault = buildH1Url(base);
  assert.equal(
    urlRoutedDefault,
    'http://127.0.0.1:8080/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=4000'
  );

  // Unmodified upstream reference
  const urlReferenceDefault = buildH1Url(base, { isReference: true });
  assert.equal(
    urlReferenceDefault,
    'http://127.0.0.1:8080/upstream/three.js/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=4000'
  );

  // WebGL backend branch
  const urlWebGL = buildH1Url(base, { backend: 'webgl', renderBundle: false, count: 1000 });
  assert.equal(
    urlWebGL,
    'http://127.0.0.1:8080/examples/webgpu_performance_renderbundle.html?backend=webgl&renderBundle=false&count=1000'
  );

  // Defect 1 Regression (Mail 7694): count=0 MUST be preserved as 0, not replaced by 4000
  const urlCountZero = buildH1Url(base, { count: 0 });
  assert.equal(
    urlCountZero,
    'http://127.0.0.1:8080/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=0'
  );
});

test('H1 Query Parser: Replicates H1 startup parameter semantics exactly, preserving count=0 and NaN', () => {
  // 1. Defaults
  const qDefault = parseH1QueryParams('');
  assert.equal(qDefault.backend, 'webgpu');
  assert.equal(qDefault.webgpu, true);
  assert.equal(qDefault.renderBundle, true);
  assert.equal(qDefault.count, 4000);

  // 2. Explicit WebGL
  const qWebGL = parseH1QueryParams('?backend=webgl&renderBundle=false&count=2500');
  assert.equal(qWebGL.backend, 'webgl');
  assert.equal(qWebGL.webgpu, false);
  assert.equal(qWebGL.renderBundle, false);
  assert.equal(qWebGL.count, 2500);

  // 3. Defect 1 Regression (Mail 7694 & 7862): ?count=0 parses to 0, not 4000
  const qZero = parseH1QueryParams('?count=0');
  assert.equal(qZero.count, 0, 'count=0 query must parse to numeric 0');

  // 4. count=NaN and count=foo preserve NaN as upstream does (Mail 7862)
  const qNaN = parseH1QueryParams('?count=NaN');
  assert.equal(Number.isNaN(qNaN.count), true, '?count=NaN must parse to NaN');

  const qFoo = parseH1QueryParams('?count=foo');
  assert.equal(Number.isNaN(qFoo.count), true, '?count=foo must parse to NaN');

  // 5. count='' (empty) falls back to 4000 via ('' || 4000)
  const qEmpty = parseH1QueryParams('?count=');
  assert.equal(qEmpty.count, 4000, '?count= must fallback to 4000');

  // 6. Full URL parsing
  const qFullUrl = parseH1QueryParams('http://localhost:8080/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=8000');
  assert.equal(qFullUrl.backend, 'webgpu');
  assert.equal(qFullUrl.count, 8000);
});

test('H1 Reload URL Computation: Matches original H1 reload() relative redirect format, preserving count=0', () => {
  const currentPath = '/examples/webgpu_performance_renderbundle.html';
  const nextUrl = computeH1ReloadUrl(currentPath, {
    backend: 'webgl',
    renderBundle: false,
    count: 4000,
  });

  assert.equal(
    nextUrl,
    '/examples/webgpu_performance_renderbundle.html?backend=webgl&renderBundle=false&count=4000'
  );

  // Defect 1 Regression (Mail 7694): reload with count: 0 emits count=0
  const nextUrlZero = computeH1ReloadUrl(currentPath, {
    backend: 'webgpu',
    renderBundle: true,
    count: 0,
  });
  assert.equal(
    nextUrlZero,
    '/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=0'
  );
});

test('Inspector Control Locator & Toggle: Interacts with Three.js Inspector DOM structures', () => {
  const eventsDispatched = [];

  const createMockRow = (labelText, isChecked) => {
    const row = {
      textContent: labelText,
      querySelector(selector) {
        if (selector === 'input[type="checkbox"]') {
          return checkbox;
        }
        return null;
      },
    };

    const checkbox = {
      type: 'checkbox',
      checked: isChecked,
      closest() { return row; },
      dispatchEvent(evt) {
        eventsDispatched.push({ type: evt.type, checked: this.checked });
      },
    };

    return row;
  };

  const rows = [
    createMockRow('render bundle', true),
    createMockRow('webgpu', true),
    createMockRow('dynamic', false),
  ];

  const mockDoc = {
    querySelectorAll(selector) {
      if (selector === '.list-item-row') return rows;
      if (selector === 'input[type="checkbox"]') return rows.map(r => r.querySelector('input[type="checkbox"]'));
      return [];
    },
    querySelector() { return null; },
  };

  // 1. Locate controls
  const bundleCtrl = findInspectorControl(mockDoc, 'render bundle');
  assert.equal(bundleCtrl.found, true);
  assert.equal(bundleCtrl.checkbox.checked, true);

  const webgpuCtrl = findInspectorControl(mockDoc, 'webgpu');
  assert.equal(webgpuCtrl.found, true);
  assert.equal(webgpuCtrl.checkbox.checked, true);

  const dynamicCtrl = findInspectorControl(mockDoc, 'dynamic');
  assert.equal(dynamicCtrl.found, true);
  assert.equal(dynamicCtrl.checkbox.checked, false);

  const nonexistent = findInspectorControl(mockDoc, 'nonexistent control');
  assert.equal(nonexistent.found, false);

  // 2. Toggle control
  const toggleRes = toggleInspectorControl(mockDoc, 'render bundle');
  assert.equal(toggleRes.success, true);
  assert.equal(toggleRes.previousValue, true);
  assert.equal(toggleRes.nextValue, false);
  assert.equal(bundleCtrl.checkbox.checked, false);
  assert.equal(eventsDispatched.length, 1);
  assert.equal(eventsDispatched[0].type, 'change');
});

test('Defect 4 Regression (Mail 7694 & 7862): detectCanvasContext operates correctly across iframe realms and rejects dummy objects', () => {
  // Mock another iframe Window realm with standard GPUCanvasContext constructor
  class GPUCanvasContext {}
  const iframeWindow = {
    GPUCanvasContext,
  };

  const iframeCanvas = {
    ownerDocument: {
      defaultView: iframeWindow,
    },
    getContext(type) {
      if (type === 'webgpu') {
        return new GPUCanvasContext();
      }
      return null;
    },
  };

  const detected = detectCanvasContext(iframeCanvas);
  assert.equal(detected, 'webgpu', 'detectCanvasContext must recognize cross-realm GPUCanvasContext');

  // Verify arbitrary JS mock without native context interface is rejected as 'none'
  const dummyCanvas = {
    getContext: () => ({ dummy: true }),
  };
  assert.equal(detectCanvasContext(dummyCanvas), 'none', 'Arbitrary JS object must not be falsely recognized as webgpu');
});

test('Snapshot Reading: Captures complete observable state from active H1 window and document', () => {
  const mockWin = {
    location: {
      href: 'http://127.0.0.1:8080/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=4000',
      pathname: '/examples/webgpu_performance_renderbundle.html',
      search: '?backend=webgpu&renderBundle=true&count=4000',
    },
    __f3d_router__: {
      getDecisionLog: () => [
        {
          site: 'WebGPURenderer',
          span: 'webgpu_performance_renderbundle.html:188:13',
          route: 'retained-upstream',
          reasons: ['specialization-unavailable'],
        },
      ],
    },
  };

  class GPUCanvasContext {}

  const mockDoc = {
    title: 'three.js webgpu - performance - renderbundle',
    querySelector(selector) {
      if (selector === 'canvas') {
        return {
          width: 800,
          height: 600,
          getContext: (type) => (type === 'webgpu' ? new GPUCanvasContext() : null),
        };
      }
      return null;
    },
    querySelectorAll(selector) {
      if (selector === '.list-item-row') {
        return [
          {
            textContent: 'render bundle',
            querySelector: () => ({ checked: true }),
          },
          {
            textContent: 'webgpu',
            querySelector: () => ({ checked: true }),
          },
          {
            textContent: 'dynamic',
            querySelector: () => ({ checked: false }),
          },
        ];
      }
      if (selector === '.inspector-log, .log-item, .item-log') {
        return [{ textContent: 'THREE.WebGPURenderer: WebGPUBackend initialized.' }];
      }
      return [];
    },
  };

  const snapshot = readH1Snapshot(mockWin, mockDoc);
  assert.equal(snapshot.pathname, '/examples/webgpu_performance_renderbundle.html');
  assert.equal(snapshot.query.backend, 'webgpu');
  assert.equal(snapshot.query.renderBundle, true);
  assert.equal(snapshot.query.count, 4000);
  assert.equal(snapshot.canvas.contextType, 'webgpu');
  assert.equal(snapshot.controls.renderBundle.checked, true);
  assert.equal(snapshot.controls.webgpu.checked, true);
  assert.equal(snapshot.controls.dynamic.checked, false);
  assert.equal(snapshot.routing.isRouted, true);
  assert.equal(snapshot.routing.activeRoute, 'retained-upstream');
  assert.deepEqual(snapshot.routing.activeReasons, ['specialization-unavailable']);
});

test('State Comparison Positive: Reference and Candidate state equality passes with honest route attribution', () => {
  // Reference (unmodified upstream)
  const refSnapshotWebGPU = {
    url: 'http://127.0.0.1:8080/upstream/three.js/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=4000',
    pathname: '/upstream/three.js/examples/webgpu_performance_renderbundle.html',
    search: '?backend=webgpu&renderBundle=true&count=4000',
    query: { backend: 'webgpu', webgpu: true, renderBundle: true, count: 4000 },
    title: 'three.js webgpu - performance - renderbundle',
    canvas: { exists: true, contextType: 'webgpu', width: 800, height: 600 },
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    inspector: { sign: 'THREE.WebGPURenderer: WebGPU' },
    routing: { isRouted: false },
  };

  // Candidate (routed facade)
  const candidateSnapshotWebGPU = {
    url: 'http://127.0.0.1:8080/examples/webgpu_performance_renderbundle.html?backend=webgpu&renderBundle=true&count=4000',
    pathname: '/examples/webgpu_performance_renderbundle.html',
    search: '?backend=webgpu&renderBundle=true&count=4000',
    query: { backend: 'webgpu', webgpu: true, renderBundle: true, count: 4000 },
    title: 'three.js webgpu - performance - renderbundle',
    canvas: { exists: true, contextType: 'webgpu', width: 800, height: 600 },
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    inspector: { sign: 'THREE.WebGPURenderer: WebGPU' },
    routing: {
      isRouted: true,
      activeRoute: 'retained-upstream',
      activeReasons: ['specialization-unavailable'],
      decisionCount: 1,
    },
  };

  const resWebGPU = compareH1Snapshots(refSnapshotWebGPU, candidateSnapshotWebGPU);
  assert.equal(resWebGPU.pass, true, `Expected pass, got diffs: ${resWebGPU.diffs.join('; ')}`);
  assert.equal(resWebGPU.diffs.length, 0);

  // WebGL branch comparison
  const refSnapshotWebGL = {
    ...refSnapshotWebGPU,
    query: { backend: 'webgl', webgpu: false, renderBundle: false, count: 2000 },
    canvas: { exists: true, contextType: 'webgl2', width: 800, height: 600 },
    controls: {
      renderBundle: { found: true, checked: false },
      webgpu: { found: true, checked: false },
      dynamic: { found: true, checked: false },
    },
  };

  const candidateSnapshotWebGL = {
    ...candidateSnapshotWebGPU,
    query: { backend: 'webgl', webgpu: false, renderBundle: false, count: 2000 },
    canvas: { exists: true, contextType: 'webgl2', width: 800, height: 600 },
    controls: {
      renderBundle: { found: true, checked: false },
      webgpu: { found: true, checked: false },
      dynamic: { found: true, checked: false },
    },
    routing: {
      isRouted: true,
      activeRoute: 'exact-backend',
      activeReasons: ['explicit-source-selection'],
      decisionCount: 1,
    },
  };

  const resWebGL = compareH1Snapshots(refSnapshotWebGL, candidateSnapshotWebGL);
  assert.equal(resWebGL.pass, true, `Expected pass, got diffs: ${resWebGL.diffs.join('; ')}`);
  assert.equal(resWebGL.diffs.length, 0);
});

test('Defect 2 Regression (Mail 7694): Rejects false-pass when both canvases are missing or contextType is none', () => {
  const refNoCanvas = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: false, contextType: 'none' }, // Both missing!
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const candidateNoCanvas = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: false, contextType: 'none' }, // Both missing!
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(refNoCanvas, candidateNoCanvas);
  assert.equal(res.pass, false, 'Must reject false-pass when canvases are uninitialized or missing');
  assert.ok(res.diffs.some(d => d.includes('Reference canvas must be initialized with valid GPU/GL context')));
  assert.ok(res.diffs.some(d => d.includes('Candidate canvas must be initialized with valid GPU/GL context')));
});

test('Defect 2 Regression (Mail 7694): Rejects false-pass when controls are missing on both', () => {
  const refNoControls = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: false, checked: false }, // Missing!
      webgpu: { found: false, checked: false }, // Missing!
      dynamic: { found: false, checked: false }, // Missing!
    },
    routing: { isRouted: false },
  };

  const candidateNoControls = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: false, checked: false }, // Missing!
      webgpu: { found: false, checked: false }, // Missing!
      dynamic: { found: false, checked: false }, // Missing!
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(refNoControls, candidateNoControls);
  assert.equal(res.pass, false, 'Must reject false-pass when controls are missing');
  assert.ok(res.diffs.some(d => d.includes('renderBundle control missing')));
  assert.ok(res.diffs.some(d => d.includes('webgpu control missing')));
  assert.ok(res.diffs.some(d => d.includes('dynamic control missing')));
});

test('Defect 3 Regression (Mail 7694): Asserts dynamic control state parity and rejects mismatch', () => {
  const ref = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false }, // false in reference
    },
    routing: { isRouted: false },
  };

  const candidate = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: true }, // true in candidate -> MISMATCH!
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(ref, candidate);
  assert.equal(res.pass, false, 'Must reject dynamic control state mismatch');
  assert.ok(res.diffs.some(d => d.includes('dynamic checked state mismatch: reference false vs candidate true')));
});

test('State Comparison Negative 1: Mismatched backend query fails comparison', () => {
  const ref = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const mutatedCandidate = {
    query: { backend: 'webgl', renderBundle: true, count: 4000 }, // Mismatch!
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(ref, mutatedCandidate);
  assert.equal(res.pass, false);
  assert.ok(res.diffs.some(d => d.includes("Backend query mismatch: reference 'webgpu' vs candidate 'webgl'")));
});

test('State Comparison Negative 2: Mismatched renderBundle checked state fails comparison', () => {
  const ref = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const mutatedCandidate = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: false }, // Mismatch!
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(ref, mutatedCandidate);
  assert.equal(res.pass, false);
  assert.ok(res.diffs.some(d => d.includes('renderBundle checked state mismatch')));
});

test('State Comparison Negative 3: Dishonest candidate routing (claiming specialized-webgpu) fails comparison', () => {
  const ref = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const dishonestCandidate = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: {
      isRouted: true,
      activeRoute: 'specialized-webgpu', // Dishonest claim!
      activeReasons: ['specialization-unavailable'],
    },
  };

  const res = compareH1Snapshots(ref, dishonestCandidate);
  assert.equal(res.pass, false);
  assert.ok(res.diffs.some(d => d.includes("Candidate route dishonest: expected 'retained-upstream', got 'specialized-webgpu'")));
});

test('State Comparison Negative 4: Polluted reference containing router fails comparison', () => {
  const pollutedRef = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true }, // Should be false for unmodified upstream!
  };

  const candidate = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(pollutedRef, candidate);
  assert.equal(res.pass, false);
  assert.ok(res.diffs.some(d => d.includes('Reference snapshot unexpectedly contains __f3d_router__')));
});

test('State Comparison Negative 5 (Mail 7862): Missing query object fails comparison', () => {
  const ref = {
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const candidate = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(ref, candidate);
  assert.equal(res.pass, false);
  assert.ok(res.diffs.some(d => d.includes('Reference snapshot missing valid query object')));
});

test('State Comparison Negative 6 (Mail 7862): Invalid contextType (e.g. 2d, undefined) fails comparison', () => {
  const ref = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: '2d' }, // Invalid context!
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const candidate = {
    query: { backend: 'webgpu', renderBundle: true, count: 4000 },
    canvas: { exists: true, contextType: '2d' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(ref, candidate);
  assert.equal(res.pass, false);
  assert.ok(res.diffs.some(d => d.includes('Reference canvas must be initialized with valid GPU/GL context')));
  assert.ok(res.diffs.some(d => d.includes('Candidate canvas must be initialized with valid GPU/GL context')));
});

test('State Comparison Negative 7 (Mail 7862): Canvas context mismatch with requested backend branch fails comparison', () => {
  // WebGL backend requested, but canvas context is webgpu
  const ref = {
    query: { backend: 'webgl', renderBundle: false, count: 2000 },
    canvas: { exists: true, contextType: 'webgpu' }, // Mismatch for webgl branch!
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: false },
      webgpu: { found: true, checked: false },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const candidate = {
    query: { backend: 'webgl', renderBundle: false, count: 2000 },
    canvas: { exists: true, contextType: 'webgpu' }, // Mismatch for webgl branch!
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: false },
      webgpu: { found: true, checked: false },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'exact-backend', activeReasons: ['explicit-source-selection'] },
  };

  const res = compareH1Snapshots(ref, candidate);
  assert.equal(res.pass, false);
  assert.ok(res.diffs.some(d => d.includes("does not match requested branch 'webgl2' for backend 'webgl'")));
});

test('State Comparison Positive (Mail 7862): Preserves count NaN parity via Object.is', () => {
  const ref = {
    query: { backend: 'webgpu', renderBundle: true, count: NaN },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: false },
  };

  const candidate = {
    query: { backend: 'webgpu', renderBundle: true, count: NaN },
    canvas: { exists: true, contextType: 'webgpu' },
    title: 'H1',
    controls: {
      renderBundle: { found: true, checked: true },
      webgpu: { found: true, checked: true },
      dynamic: { found: true, checked: false },
    },
    routing: { isRouted: true, activeRoute: 'retained-upstream', activeReasons: ['specialization-unavailable'] },
  };

  const res = compareH1Snapshots(ref, candidate);
  assert.equal(res.pass, true, `Expected NaN count parity to pass via Object.is, got: ${res.diffs.join('; ')}`);
});

test('Pixel Comparison Positive: Identical pixel buffers pass with 0.0% diff and zero RMSE', () => {
  const width = 100;
  const height = 100;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 193;     // R (H1 background 0xc1c1c1)
    data[i + 1] = 193; // G
    data[i + 2] = 193; // B
    data[i + 3] = 255; // A
  }

  const ref = { width, height, data };
  const cand = { width, height, data: new Uint8ClampedArray(data) };

  const res = comparePixelBuffers(ref, cand);
  assert.equal(res.pass, true);
  assert.equal(res.diffPixels, 0);
  assert.equal(res.diffPercent, 0);
  assert.equal(res.rmse, 0);
  assert.equal(res.maxChannelDiff, 0);
});

test('Pixel Comparison Positive: Pixel delta within colorTolerance (2) passes', () => {
  const width = 100;
  const height = 100;
  const refData = new Uint8ClampedArray(width * height * 4).fill(128);
  const candData = new Uint8ClampedArray(width * height * 4).fill(130); // delta = 2 <= tolerance 2

  const ref = { width, height, data: refData };
  const cand = { width, height, data: candData };

  const res = comparePixelBuffers(ref, cand, { colorTolerance: 2 });
  assert.equal(res.pass, true);
  assert.equal(res.diffPixels, 0);
  assert.equal(res.maxChannelDiff, 2);
});

test('Pixel Comparison Negative (Mail 13025): Mismatched pixel buffer exceeding 0.1% threshold strictly rejected', () => {
  const width = 100;
  const height = 100;
  const totalPixels = width * height; // 10,000 pixels
  const refData = new Uint8ClampedArray(totalPixels * 4).fill(100);

  const ref = { width, height, data: refData };

  // Mutate 1.0% of pixels (100 pixels) using createMismatchedPixelBuffer
  const mutated = createMismatchedPixelBuffer(ref, 1.0);

  const res = comparePixelBuffers(ref, mutated, { colorTolerance: 2, maxDiffPixelPercent: 0.1 });
  assert.equal(res.pass, false, 'Mutated pixels exceeding 0.1% must fail');
  assert.ok(res.diffPercent > 0.1, `diffPercent ${res.diffPercent}% must exceed 0.1% threshold`);
  assert.ok(res.diffPixels >= 10, `diffPixels ${res.diffPixels} must be non-zero`);
  assert.ok(res.rmse > 0, `rmse ${res.rmse} must be positive`);
});

test('Pixel Comparison Negative: Dimension mismatch is rejected', () => {
  const ref = { width: 100, height: 100, data: new Uint8ClampedArray(100 * 100 * 4) };
  const cand = { width: 200, height: 100, data: new Uint8ClampedArray(200 * 100 * 4) };

  const res = comparePixelBuffers(ref, cand);
});

test('Canvas Capture: Uses native canvas dimensions without downscaling or resampling (ROOT H1 REVIEW)', () => {
  let drawnArgs = null;
  const mockCanvas = {
    width: 800,
    height: 600,
    ownerDocument: {
      createElement(tag) {
        if (tag !== 'canvas') return null;
        return {
          width: 0,
          height: 0,
          getContext(type) {
            if (type !== '2d') return null;
            return {
              drawImage(...args) {
                drawnArgs = args;
              },
              getImageData(x, y, w, h) {
                return {
                  width: w,
                  height: h,
                  data: new Uint8ClampedArray(w * h * 4),
                };
              },
            };
          },
        };
      },
    },
  };

  const captured = captureCanvasPixels(mockCanvas);
  assert.equal(captured.width, 800, 'Native width preserved');
  assert.equal(captured.height, 600, 'Native height preserved');
  assert.equal(captured.data.length, 800 * 600 * 4, 'Full native pixel buffer size');
  // Assert drawImage was called with exact 1:1 coordinates (no resampling)
  assert.deepEqual(drawnArgs.slice(1), [0, 0, 800, 600, 0, 0, 800, 600], '1:1 native canvas blit');
});

test('Pixel Comparison Field Parity: Returns both .pass and .passed booleans consistently (ROOT H1 REVIEW)', () => {
  const buf = { width: 10, height: 10, data: new Uint8ClampedArray(10 * 10 * 4).fill(128) };
  const res = comparePixelBuffers(buf, buf);
  assert.equal(res.pass, true);
  assert.equal(res.passed, true);

  const mutated = createMismatchedPixelBuffer(buf, 5.0);
  const failRes = comparePixelBuffers(buf, mutated);
  assert.equal(failRes.pass, false);
  assert.equal(failRes.passed, false);
});

test('Seeded Random Prelude: Injects deterministic PRNG before application scripts in HTML (ROOT H1 REVIEW)', () => {
  const sampleHtml = `<!DOCTYPE html><html><head><script type="importmap">{"imports":{}}</script></head><body><script type="module">console.log("app");</script></body></html>`;
  const injected = injectSeededRandomPrelude(sampleHtml);
  assert.ok(injected.includes('id="f3d-seeded-random-prelude"'));
  assert.ok(injected.indexOf('<script type="importmap">') < injected.indexOf('id="f3d-seeded-random-prelude"'));
  assert.ok(injected.indexOf('id="f3d-seeded-random-prelude"') < injected.indexOf('<script type="module">'));
  // Repeated injection is idempotent
  const twice = injectSeededRandomPrelude(injected);
  assert.equal(twice, injected);
});

test('Seeded Random Sequence: Produces identical float sequences across multiple independent initializations (ROOT H1 REVIEW)', () => {
  function makeRng(seed = 0x12345678) {
    let s = seed;
    return function() {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const rng1 = makeRng();
  const rng2 = makeRng();

  const seq1 = Array.from({ length: 100 }, () => rng1());
  const seq2 = Array.from({ length: 100 }, () => rng2());

  assert.deepEqual(seq1, seq2, 'Both RNG instances must produce identical deterministic sequence');
  assert.ok(seq1[0] >= 0 && seq1[0] < 1, 'Uniform float in [0, 1)');
});

test('Checkpoint Observations: Extracts read-only observations without state copying (ROOT H1 REVIEW)', () => {
  const mockWin = {
    __f3d_rng_count__: 42,
    __f3d_rng_state__: 12345,
    __f3d_last_scene__: {
      children: [
        {
          isGroup: true,
          children: [
            {
              position: { x: 1, y: 2, z: 3 },
              quaternion: { x: 0, y: 0, z: 0, w: 1 },
              scale: { x: 0.5, y: 0.5, z: 0.5 },
              material: { color: { getHexString: () => 'ff00ff' } },
              matrix: { elements: new Float32Array(16) },
            },
          ],
        },
      ],
    },
    __f3d_last_camera__: {
      position: { x: 0, y: 0, z: 50 },
      quaternion: { x: 0, y: 0, z: 0, w: 1 },
      projectionMatrix: { elements: new Float32Array(16) },
      aspect: 1.333,
      fov: 70,
      near: 1,
      far: 100,
    },
  };

  const obs = extractCheckpointObservations(mockWin, {});
  assert.ok(obs);
  assert.equal(obs.rng.count, 42);
  assert.equal(obs.rng.state, 12345);
  assert.deepEqual(obs.firstMesh.position, [1, 2, 3]);
  assert.equal(obs.firstMesh.colorHex, 'ff00ff');
  assert.deepEqual(obs.camera.position, [0, 0, 50]);

  // Comparison with identical passes
  const compPass = compareCheckpointObservations(obs, obs);
  assert.equal(compPass.pass, true);
  assert.equal(compPass.diffs.length, 0);

  // Comparison with mismatched RNG fails
  const mismatchedRng = JSON.parse(JSON.stringify(obs));
  mismatchedRng.rng.count = 43;
  const compRngFail = compareCheckpointObservations(obs, mismatchedRng);
  assert.equal(compRngFail.pass, false);
  assert.ok(compRngFail.diffs.some(d => d.includes('RNG invocation count mismatch')));

  // Comparison with mismatched color fails
  const mismatchedColor = JSON.parse(JSON.stringify(obs));
  mismatchedColor.firstMesh.colorHex = '00ff00';
  const compColorFail = compareCheckpointObservations(obs, mismatchedColor);
  assert.equal(compColorFail.pass, false);
  assert.ok(compColorFail.diffs.some(d => d.includes('First mesh color mismatch')));
});
