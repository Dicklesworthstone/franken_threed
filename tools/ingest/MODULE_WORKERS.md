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
`new URL` assets are emitted beneath a content-addressed `f3d-workers/` directory.
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

The graph is bounded to 4096 module instances, 4096 assets and 64 MiB of input or
relocated output. The plugin also enforces shared limits across worker entries
in one build. Missing dependencies, unsupported open dynamic imports and budget
violations fail rather than emit an apparently runnable partial graph. Each
worker graph is fully planned before it is emitted. Module and binary changes
change the emitted directory hash, including transitive dependencies. Watch
inputs are registered and graph emission state is reset at each Rollup build.

## Boundaries

Construction must use the direct `Worker`/`SharedWorker` and `URL` names with an
explicit, plain object `type: 'module'`. A binding or write to a constructor name
anywhere in that source module conservatively prevents this substitution.
Indirection, computed options, spreads and classic-worker conversion are not
claimed. Arbitrary networking, generated code/DOM, dynamic asset names and
service workers are not closed by this pass. For local fetches, use
`fetch(new URL('./data.bin', import.meta.url))`; plain relative fetch literals are
rejected because their base is the worker entry, not the importing module.

These are multi-file application artifacts. The separate single-HTML exporter
retains its existing worker restrictions (including cyclic graphs and nested
workers); this feature does not override those checks or weaken CSP.

## Checks

```sh
node --test tests/unit/module_workers.test.mjs
```

The suite executes emitted ESM in actual Node worker threads with a small host
adapter for browser messaging. It covers nested workers, transferable buffers,
cycles, lazy imports, instance identity, binary assets, deterministic output,
constructor admission and resource limits. The plugin-hook execution tests call
its real `transform` and `resolveFileUrl` methods with recorded Rollup emissions.
They do not substitute for full Rollup/application-build or browser tests.
