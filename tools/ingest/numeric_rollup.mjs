/**
 * Post-link numeric specialization for real application bundles. Running after
 * Rollup links each ES chunk exposes closed update functions and their callers
 * across source-module boundaries without replacing public function exports.
 * Static runtime assets are emitted only when a kernel is admitted.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import * as acorn from 'acorn';
import { specializeNumericModule } from './numeric_specialization.mjs';

const REPORT_FILE = 'f3d-numeric-specialization.json';
const hash = source => crypto.createHash('sha256').update(source).digest('hex');
const relativeImport = (from, to) => {
  const relative = path.posix.relative(path.posix.dirname(from), to);
  return relative.startsWith('.') ? relative : './' + relative;
};

/** @returns {import('rollup').Plugin} */
export function numericKernelRollupPlugin(options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('Numeric specialization options must be an object');
  for (const key of Object.keys(options)) {
    if (!['maxKernels', 'maxMemoryPages'].includes(key)) throw new TypeError(`Unknown numeric specialization option: ${key}`);
  }
  // Validate budgets even for applications with no candidates.
  specializeNumericModule('', options);
  const settings = { ...options };
  let units, assets, finalReport;

  function runtime(context) {
    if (assets.size) return [...assets.keys()][1];
    const runtimeSource = fs.readFileSync(new URL('./numeric_kernel_runtime.mjs', import.meta.url), 'utf8');
    const runtimeName = `f3d-runtime/numeric-kernel-${hash(runtimeSource).slice(0, 20)}.mjs`;
    let dispatcher = fs.readFileSync(new URL('./numeric_dispatch.mjs', import.meta.url), 'utf8');
    const ast = acorn.parse(dispatcher, { ecmaVersion: 'latest', sourceType: 'module' });
    const imports = ast.body.filter(node => node.type === 'ImportDeclaration');
    if (imports.length !== 1 || imports[0].source.value !== './numeric_kernel_runtime.mjs') {
      throw new Error('Numeric dispatch runtime dependency contract changed');
    }
    const literal = imports[0].source;
    dispatcher = dispatcher.slice(0, literal.start) + JSON.stringify('./' + path.posix.basename(runtimeName)) + dispatcher.slice(literal.end);
    const dispatchName = `f3d-runtime/numeric-dispatch-${hash(dispatcher).slice(0, 20)}.mjs`;
    for (const [fileName, source] of [[runtimeName, runtimeSource], [dispatchName, dispatcher]]) {
      context.emitFile({ type: 'asset', fileName, source });
      assets.set(fileName, source);
    }
    return dispatchName;
  }

  return {
    name: 'f3d-numeric-specialization',
    api: { getReport() { return finalReport; } },
    renderStart() { units = new Map(); assets = new Map(); finalReport = null; },
    renderChunk(code, chunk, output) {
      let result;
      if (output.format !== 'es' || output.sourcemap) {
        // Preserve the application's requested output contract. This pass does
        // not yet emit source maps and must not silently corrupt existing ones.
        result = { changed: false, report: {
          version: 1, sourceName: chunk.name, route: 'retained-js', accelerated: false,
          compiledKernels: 0, rewrittenCalls: 0, candidates: [],
          refusal: { code: output.format !== 'es' ? 'NON_ES_OUTPUT' : 'SOURCE_MAP_SPECIALIZATION_UNAVAILABLE' },
        } };
      } else {
        result = specializeNumericModule(code, {
          ...settings, sourceName: chunk.name,
          runtimeModule: () => relativeImport(chunk.fileName, runtime(this)),
        });
      }
      units.set(chunk.fileName, {
        ...result.report,
        coordinateSpace: 'renderChunk-before-specialization-and-hash-substitution',
        inputSha256: hash(code), moduleIds: [...(chunk.moduleIds ?? Object.keys(chunk.modules))],
      });
      if (!result.changed) return null;
      const dispatchName = runtime(this);
      // Rollup explicitly requires renderChunk plugins to update this metadata
      // when adding imports; downstream emitters must see the actual dependency.
      if (!chunk.imports.includes(dispatchName)) chunk.imports.push(dispatchName);
      chunk.importedBindings[dispatchName] = ['createNumericDispatch', 'dispatchNumericCall'];
      return { code: result.code, map: null };
    },
    generateBundle(_output, bundle) {
      const chunks = Object.values(bundle).filter(item => item.type === 'chunk');
      const reports = [...units].map(([preliminaryFileName, report]) => {
        const chunk = chunks.find(item => item.preliminaryFileName === preliminaryFileName || item.fileName === preliminaryFileName);
        if (!chunk) throw new Error(`Numeric specialization report lost its output chunk: ${preliminaryFileName}`);
        return { ...report, fileName: chunk.fileName };
      }).sort((a, b) => a.fileName.localeCompare(b.fileName, 'en'));
      // Explicit content-addressed names must not be shadowed by another output.
      for (const [fileName, source] of assets) {
        const asset = bundle[fileName];
        if (asset?.type !== 'asset' || Buffer.from(asset.source).compare(Buffer.from(source)) !== 0) {
          throw new Error(`Numeric runtime asset collision: ${fileName}`);
        }
      }
      if (Object.hasOwn(bundle, REPORT_FILE)) throw new Error(`Numeric report asset collision: ${REPORT_FILE}`);
      finalReport = {
        version: 1, enabled: true, accelerated: false,
        scope: 'guarded-direct-calls-within-linked-es-chunks',
        compiledKernels: reports.reduce((sum, report) => sum + report.compiledKernels, 0),
        rewrittenCalls: reports.reduce((sum, report) => sum + report.rewrittenCalls, 0),
        runtimeAssets: [...assets].map(([fileName, source]) => ({ fileName, sha256: hash(source) })),
        units: reports, reportFile: REPORT_FILE,
      };
      this.emitFile({ type: 'asset', fileName: REPORT_FILE, source: JSON.stringify(finalReport, null, 2) + '\n' });
    },
  };
}
