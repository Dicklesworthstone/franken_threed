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

## Native JSON modules and finite URL choices

Inside a worker module, JSON imports and reexports retain their native module
goal, rather than becoming JavaScript object-literal wrappers:

```js
import settings from './settings.json' with { type: 'json' };
const same = await import(new URL('./settings.json', import.meta.url), {
  with: { type: 'json' },
});
```

The emitted JSON bytes are unchanged. Native parsing preserves default-only
exports, shared mutable identity for repeated typed imports, duplicate-key and
signed-zero behavior, and lazy rejection of malformed or unselected JSON.
Query/fragment variants and different worker realms retain distinct identities.
The type attribute, not a guessed file extension, selects the module goal.
Local dynamic import options require a statically known plain attribute
dictionary; unknown types, spreads, accessors and computed keys are not guessed.
Options and their admitted sequence-expression side effects remain in place.
Serve emitted `.json` resources with the appropriate JSON MIME type.

Worker construction, dynamic imports and URL assets can select from bounded
literal alternatives at runtime:

```js
const decoder = new Worker(new URL(
  useFastDecoder ? './fast.mjs' : './portable.mjs', import.meta.url,
), { type: 'module' });
```

Both entry graphs are packaged, but only the original selected constructor runs.
Conditions and comma-expression prefixes execute once in their original order.
Only result operands are rewritten; no selector evaluation, eager import,
dispatch wrapper or production loader is added. Analysis is bounded to 128
alternatives and bounded recursion; a partial set is never treated as closed.

`import(new URL(..., import.meta.url))` uses URL resolution, even for paths
without a leading `./`, rather than package/import-map resolution. Such imports
are executable dependencies, not raw asset copies. ESM helpers in classic
workers can similarly pass URL objects to native `importScripts`; these resolve
from the helper's module URL, while string arguments still use the worker entry.
Origin-root/protocol-relative URL operands remain host-owned.

Conditional plain-string child workers are admitted inside known worker realms.
Plain document-relative worker strings are not relocated: arbitrary document
base mutations cannot be replaced with a build-time guessed base. Unknown
constructor bindings, open URL expressions and reflective host changes are not
claimed closed by this bounded analysis.

## Validation

```sh
node --test tests/unit/classic_workers.test.mjs \
  tests/unit/classic_workers_execution.test.mjs \
  tests/unit/module_workers.test.mjs \
  tests/unit/worker_json_modules.test.mjs \
  tests/unit/worker_url_choices.test.mjs
```

The Node execution suite compares original and emitted source in actual worker
threads. V8 evaluates classic scripts with the Script goal, and Node loads ESM
with its native linker. A test-only host adapter supplies local importScripts,
file Responses and browser-style messaging. Tests cover mixed nested threads,
shared-script entry-relative resources, streaming compilation of real Wasm,
native transferable ArrayBuffers, lexical redeclaration errors, repeated and
recursive script evaluation, and a raw-copy failure regression.

The JSON/choice cases additionally compare execution after the source directory
has been renamed out of reach. They exercise native JSON identity and errors,
conditional worker selection, URL-object imports, and option/selector ordering.
The Rollup-plugin checks invoke production hooks and execute the resulting parent
module; they do not substitute for running the full Rollup bundling engine.

A native Chromium HTTP run was attempted on October 13, but the managed browser
blocked navigation before either application executed with
`net::ERR_BLOCKED_BY_ADMINISTRATOR`. No browser pass is claimed and no browser
policy was changed. Node results do not establish browser HTTP/CORS/MIME, CSP,
Trusted Types or SharedWorker lifetime compatibility. Full application/Rollup
integration and the full repository suites remain separate validation gates.

The single-HTML exporter remains a separate path with its existing restrictions.
This implementation is retained JavaScript packaging, not Rust specialization or
a measured performance improvement.
