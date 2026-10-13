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
substitution and numeric specialization are not applied across this realm
boundary. Explicit HTTP/HTTPS/data imports stay external; the compiler does not
fetch them. Origin-root and other non-file worker/asset URLs stay host-owned.

The graph is bounded to 4096 realm-specific module instances, 4096 assets and
64 MiB of physical inputs, realm-expanded source work or relocated output. The plugin also enforces shared limits across worker entries
in one build. Missing dependencies, unsupported open dynamic imports and budget
violations fail rather than emit an apparently runnable partial graph. Each
worker graph is fully planned before it is emitted. Module and binary changes
change the emitted directory hash, including transitive dependencies. Watch
inputs are registered and graph emission state is reset at each Rollup build.
Nested closure hashing has its own work bound. Shared build accounting is
conservative: overlapping graph plans can be charged again even when identical
emitted files are reused.

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
