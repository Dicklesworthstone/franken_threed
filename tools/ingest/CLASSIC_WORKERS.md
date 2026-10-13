# Classic and mixed worker graphs

The application bundler's worker pass now packages classic workers as executable
script graphs, not opaque URL assets. Module-worker behavior remains available.
The existing `moduleWorkerRollupPlugin` name is retained for callers; it now
handles both worker types when constructed from an explicit module-relative URL:

```js
const decoder = new Worker(new URL('./decoder.js', import.meta.url));
const explicit = new Worker(new URL('./decoder.js', import.meta.url), {
  type: 'classic', name: 'geometry-decoder',
});
```

Omitted options, an empty/plain dictionary with no type, and `null` retain the
classic default. The legacy SharedWorker string-name overload is also admitted.
Computed/spread/accessor options and unknown types are not guessed. Constructors,
options, messaging, error handling and lifetime stay host-owned. No production
script evaluator, importScripts shim, fetch shim or Worker wrapper is shipped.

## Executable closure without changing the script goal

Classic entries and their `importScripts` dependencies remain ordinary `.js`
classic scripts. The packager changes only known URL operands. It does not add
ESM wrappers, strict mode, eager concatenation or execution deduplication.
Synchronous multi-argument order, repeated script execution, global var/function
and lexical declarations, exceptions and guarded recursive imports stay with
the host. A script URL may have both classic and ESM records without conflating
their execution goals.

Direct `importScripts`, `self.importScripts` and `globalThis.importScripts`
accept literal/static-template arguments and conditionals with literal branches.
Every selector stays in the original expression and is evaluated by the host.
An open script expression or missing local dependency fails before graph
emission. HTTP/HTTPS/data and origin-root script URLs remain host-owned.
Calling importScripts in a module worker is not converted into a working loader:
the original native TypeError path remains, without reading its script URLs.

Both `importScripts('./helper.js')` and `fetch('./data.bin')` use the **worker
entry** as their base, including calls inside imported scripts. Dynamic ESM
imports instead resolve from the script/module containing the import. Fetches
retain the existing static GET admission and byte limits. One helper used by two
worker entries receives two context-specific closures with the correct resources.

Inside a worker, direct literal `new Worker('./child.js')` constructors are also
packaged using that worker's entry base. Nested classic and module workers can
alternate arbitrarily within existing graph limits. ESM lazy imports, reexports,
cycles, live bindings and top-level await continue using native modules.

Worker kind is part of the realm key. A classic and module worker constructed
from the same source URL receive separate entry references. Nested shared entry
addresses remain independent of unrelated parent code. Script instances count
against maxModules; source expansion, assets, output and closure analysis retain
the existing bounds. Shared host-name rebinding across classic scripts is
conservatively rejected when it would make a transformed host call ambiguous.
This is not whole-program analysis of reflective global mutation or arbitrary
runtime URL construction.

## Validation

```sh
node --test tests/unit/classic_workers.test.mjs \
  tests/unit/classic_workers_execution.test.mjs
```

The Node execution suite compares original and emitted source in actual worker
threads. V8 evaluates classic scripts with the Script goal, and Node loads ESM
with its native linker. A test-only host adapter supplies local importScripts,
file Responses and browser-style messaging. Tests cover mixed nested threads,
shared-script entry-relative resources, streaming compilation of real Wasm,
native transferable ArrayBuffers, lexical redeclaration errors, repeated and
recursive script evaluation, and a raw-copy failure regression.

A native Chromium HTTP run was attempted on October 13, but the managed browser
blocked navigation before either application executed with
`net::ERR_BLOCKED_BY_ADMINISTRATOR`. No browser pass is claimed and no browser
policy was changed. Node results do not establish browser HTTP/CORS/MIME, CSP,
Trusted Types or SharedWorker lifetime compatibility. Full application/Rollup
integration and the full repository suites remain separate validation gates.

The single-HTML exporter remains a separate path with its existing restrictions.
This implementation is retained JavaScript packaging, not Rust specialization or
a measured performance improvement.
