# Module workers in application builds

`bundleWithRollup` (and therefore `--build-app`) packages explicit module workers
before treating `new URL(...)` expressions as ordinary assets:

```js
const simulation = new Worker(new URL('./simulation.mjs', import.meta.url), {
  type: 'module',
  name: 'simulation',
});
```

The same direct constructor pattern is supported for `SharedWorker`. The worker
entry, static imports/reexports, literal dynamic imports, conditionals whose
branches are literal imports, nested module workers, and module-relative
`new URL` assets are emitted beneath content-addressed `f3d-workers/` directories.
All worker modules remain native ESM; cycles, top-level await, live bindings,
separate query/fragment instances and lazy evaluation are not reimplemented.
Native constructors continue to own messaging, transferable buffers, isolation,
SharedWorker naming and lifetime. Options are not evaluated or changed at build
time. No worker is executed by the compiler.

Worker imports use installed-package resolution without inheriting the page's
import map. The existing explicit Three.js package override is retained. Renderer
substitution is not applied across this realm boundary. Numeric specialization
is opt-in, as described below. Explicit HTTP/HTTPS/data imports stay external; the compiler does not
fetch them. Origin-root and other non-file worker/asset URLs stay host-owned.

Worker lazy imports may select paths imported from configuration modules or
installed ESM packages. The same bounded primitive-string proof used by the
document compiler follows immutable named/default exports and explicit
reexports, without evaluating configuration code:

```js
import { first, second } from './routes.mjs';
const load = useFirst => import(useFirst ? first : second);
```

Both targets are included in normal emitted worker graphs and remain packable
into standalone HTML. The selected strings resolve from the consuming worker
module, independently of the configuration module's location. Initializers,
selectors, top-level await, lazy evaluation, query identities and JSON import
options keep their native order and identity. Reads share the worker frontend
snapshot and physical-input budgets; all inspected dependencies remain watched.
Invalid source, frontend failures and exhausted worker budgets keep their error
codes. Mutable bindings, namespace selections, star reexports, opaque values and
unresolved imports retain the existing open-worker boundary. This extension
applies to `import()`; worker constructors, fetch and other resource consumers
keep their existing local-selector requirements.

The graph is bounded to 4096 realm-specific module instances, 4096 assets and
64 MiB of physical inputs, realm-expanded source work or relocated output. The plugin also enforces shared limits across worker entries
in one build. Missing dependencies, unsupported open dynamic imports and budget
violations fail rather than emit an apparently runnable partial graph. Each
worker graph is fully planned before it is emitted. Module and binary changes
change the emitted directory hash, including transitive dependencies. Watch
inputs are registered and graph emission state is reset at each Rollup build.
Standalone plugins also replace their owned frontend source snapshot at
`buildStart()`, so dependency edits reach the next worker output and numeric
report. All worker realms within one build still normalize each source URL and
kind once. An explicitly supplied `_moduleSources` snapshot belongs to its
caller: the worker plugin preserves records already ingested by other passes.
The application build creates a fresh shared snapshot for each build call.
Invalid module syntax, including failures during early module classification or
numeric compilation, reports `F3D_WORKER_PARSE` with the failing module URL;
frontend failures and unsupported CommonJS retain their own error codes.
Nested closure hashing has its own work bound. Shared build accounting is
conservative: overlapping graph plans can be charged again even when identical
emitted files are reused.

## Node worker threads

Node application builds (`--target node` or `{target: 'node'}`) close native ESM
worker entries imported from `node:worker_threads` or `worker_threads`:

```js
import { Worker as Thread } from 'node:worker_threads';
const simulation = new Thread(new URL('./simulation.mjs', import.meta.url), {
  workerData: input,
  transferList: [input.buffer],
});
```

Named import aliases and namespace imports are recognized with their lexical
scope. Nested Node workers use the same closure. Static and bounded lazy ESM,
Node package conditions, native builtins, JSON modules and module-relative file
assets are emitted through the existing worker graph. Node's filesystem entry
rules remove worker URL query/fragment suffixes and resolve symlinks; imported
ESM query/fragment instances remain distinct. `buildWorkerGraph` and
`moduleWorkerRollupPlugin` also accept `target: 'node'` directly.

The native constructor and every option expression remain in place. Node owns
workerData cloning, transfer lists, environment and argument handling, events,
errors, termination and each worker realm. No worker, messaging or fetch shim is
added. Relative fetch strings retain Node's native behavior, while explicit
`new URL('./data.bin', import.meta.url)` resources are relocated for file APIs.
Missing worker entries retain native asynchronous errors. No browser renderer
or mandatory GPU/Wasm startup is introduced; existing numeric specialization
remains opt-in.

This closure uses the same ordinary Node module conditions as the application
target. Explicit runtime `execArgv` or alternate loader, condition and symlink
flags require a matching frontend and receive a build diagnostic; benign literal
flags remain native. Opaque options must use this same default resolution profile.
The mutable default-object `threads.Worker` form also receives a diagnostic;
use a named or namespace import, or normalize it in the project frontend.
CommonJS worker graphs and unresolved module-relative entry/import choices keep
an explicit build boundary. CWD-relative filename strings, generated/eval
workers and opaque constructor origins remain native host routes and are not
claimed as closed by this pass.

## Numeric Wasm inside module workers

The application compiler's existing numeric opt-in also applies to local ESM
inside worker graphs, including module workers launched by classic page scripts:

```sh
node tools/ingest/cli.mjs app.html --build-app dist --specialize-numeric
```

The Node API accepts the same `specializeNumeric: true` or numeric option object.
Eligible functions, cross-module calls and closed loop islands use the existing
guarded Wasm compiler. Each worker realm loads its own lazy dispatcher; source
code still owns messaging, transfers, asynchronous boundaries and public function
identities. Missing Wasm, failed guards and work-budget exhaustion select the
existing source implementation without publishing partial numeric updates.

Compiler runtime modules join the worker's normal content-addressed closure and
size limits. Their bytes and the numeric settings affect emitted identities;
application frontend transforms and resolvers cannot rewrite those compiler
inputs. Explicit opt-out preserves the ordinary worker output. The build report
includes worker-specific units and adds their kernel/call counts to application
totals; those counts describe generated code, not executed calls or speedup.

`node --test tools/ingest/numeric_workers.integration.test.mjs` checks actual
isolated Node worker threads, transferable and overlapping typed arrays, lazy
imports, module identity, real Wasm calls, fallback, relocated application builds
and packed worker modules. It also rebuilds edited kernels with a reused plugin
and executes the new Wasm in a fresh worker. Browser Worker/SharedWorker host behavior retains its
separate validation requirements. Classic worker scripts themselves remain
ordinary JavaScript; their imported ESM can use this numeric path.

## Worker resource loading

Static relative GET requests are closed along with the worker's module graph:

```js
const settings = await (await fetch('./settings.json')).json();
const { instance } = await WebAssembly.instantiateStreaming(fetch('./solver.wasm'));
```

Literal strings, static templates and conditionals with literal branches work
with direct `fetch`, `globalThis.fetch` and `self.fetch` calls. The initializer
must omit the method or specify a literal GET in a plain object; an unknown
initializer, spread, accessor or non-GET request fails instead of becoming an
accidental local-file read. Options and conditional selectors are left in place
and evaluated once by the host, in their original order. Remote, origin-root and
dynamic networking are left unchanged and are not claimed closed. Serve `.wasm`
resources with `application/wasm` for browser streaming compilation.

A plain relative fetch resolves from the **worker entry URL**, even when its call
lives in an imported helper. In contrast, `new URL('./data.bin', import.meta.url)`
is module-relative. The packager distinguishes these bases and preserves each
asset's query/fragment suffix. A helper reached from two different nested worker
entries receives two realm-specific module instances with the correct asset
references; repeated imports within one realm still share one module instance.
There is no production fetch or messaging shim.

Each worker entry's directory hash depends on that entry's transitive closure,
not an unrelated discovering parent's source. Thus identical nested entry URLs
can be reused across graph plans instead of accidentally changing constructor
identity. A child dependency or resource change invalidates both that child's
address and its dependent parents. Mutual worker-construction graphs have a
finite symbolic hash; the compiler does not execute the constructors. The
plugin reuses already emitted nested entries and identical asset files.
This preserves URLs, not host capabilities: it does not expose SharedWorker in
environments where that constructor is unavailable.

## Boundaries

Construction must use the direct `Worker`/`SharedWorker` and `URL` names with an
explicit, plain object `type: 'module'`. A binding or write to a constructor name
anywhere in that source module conservatively prevents this substitution.
Indirection, computed options, spreads and classic-worker conversion are not
claimed. Arbitrary networking, generated code/DOM, dynamic asset names and
service workers are not closed by this pass. Static fetch handling does not
infer URLs hidden behind Request objects or arbitrary variable expressions.
Resource relocation changes observable source paths and response URLs, just as
the ordinary application build does; it does not emulate an original server's
routing, response headers, redirects or source-file introspection.

These are multi-file application artifacts. The separate single-HTML exporter
retains its existing worker restrictions (including cyclic graphs and nested
workers); this feature does not override those checks or weaken CSP.

## Checks

```sh
node --test tests/unit/module_workers.test.mjs
```

The suite executes emitted ESM in actual Node worker threads with a small host
adapter for browser messaging and local file responses. It covers nested
workers, transferable buffers, native Wasm streaming compilation, JSON/text
fetches, distinct entry-relative fetch bases, cycles, lazy imports, instance
identity, binary assets, deterministic output, constructor admission and resource
limits. The test file-response adapter supplies Wasm MIME metadata; it does not
prove browser HTTP, CORS or MIME behavior. SharedWorker checks record and compare
constructor URLs, not native SharedWorker connections or lifetime. The
plugin-hook execution tests call the real `transform` and `resolveFileUrl`
methods with recorded Rollup emissions. They do not substitute for full
Rollup/application-build or browser tests.
