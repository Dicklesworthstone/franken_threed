# Native modules in single-HTML exports

`packHtml` closes the static module graph of an already emitted application.
The ordinary multi-file build is not modified. This is retained-JavaScript
packaging (F3D-10), not a renderer, compiler specialization or speedup claim.

## Typed imports and source errors

Static imports, reexports and bounded lazy imports retain their original import
attributes. Document and worker module identities include the requested type as
well as the source URL, including query/fragment variants. JSON bytes are copied
unchanged: no object-literal wrapper, eager JSON parsing, BOM removal or numeric
round trip occurs during packing. An unselected malformed JSON dependency does
not fail the build; the native importer reports its error when selected.

Requested type and response MIME are separate. The emitted-file MIME table is
used, not an assumption that `type: 'json'` authorizes changing JavaScript into
JSON. Missing JSON attributes, incompatible types, named JSON exports and
unsupported extra attributes remain native errors. Files with unknown MIME
suffixes require the ordinary build (`MODULE_MIME`); custom server MIME rules
are not inferred from file contents. Module-worker entries require JavaScript
MIME before a Blob entry is generated.

Dynamic import options require a statically known plain dictionary, including a
plain `with` dictionary and a literal JSON type where present. Sequence-result
expressions are admitted without evaluating or removing their preceding effects.
Spreads, computed/accessor properties and unknown types fail before publication.
The original options and extra attributes are retained for native evaluation.

## URL-object imports

`import(new URL('./decoder.mjs', import.meta.url))` is an executable dependency,
not a raw asset copy. Literal conditional choices and sequence-result operands
are supported; selectors, URL construction and option effects remain in place.
URL operands use URL resolution, not bare-package or document import-map rules.

Workers use absolute embedded data-module URLs and never inherit document maps.
Document URL operands use absolute `f3d-packed:` aliases in the generated import
map, allowing native document cycles and live bindings to close without recursive
data-URL expansion. The aliases and ordinary import keys resolve to the same
module URL. Worker module cycles still fail explicitly rather than shipping a
replacement linker. Source `import.meta` reflection outside admitted static URL
operands remains unsupported, as do reflective changes to host constructors.

## Nested dedicated worker graphs

Dedicated workers may be constructed by document modules, module workers,
classic workers and ESM helpers inside classic workers. Finite literal,
conditional and sequence-result URLs are closed without executing selectors.
Native URL-object arguments stay URL objects. Omitted, null and empty/plain
options retain the classic default; type/name and option-expression effects
remain in the original constructor call. No Worker wrapper is emitted.

Each worker dependency graph records its entry URL and classic/module kind.
String worker URLs, `importScripts` and static GET fetches resolve from that
entry, including in shared helper modules. Explicit `new URL(...,import.meta.url)`
operands and ESM imports instead resolve from their containing module. Shared
helpers therefore receive separate closures where the worker environment differs.
JSON identities, source MIME, lazy imports and module-local bindings are retained.

Classic scripts are not concatenated, wrapped in ESM or made strict. A classic
entry that needs nested-worker factories receives private global lexical
declarations after its original directive prologue. Imported scripts share these
bindings without redeclaring them on repeated `importScripts` calls. Factory name
collisions and direct host-name rebinding across the classic closure fail before
output. Factories allocate Blob URLs lazily and cache them; they do not expose
new properties on the global object. The separate classic and ESM factory forms
have separate bounded caches. This does not establish cross-realm SharedWorker
identity or lifetime semantics; SharedWorker remains unsupported here.

Classic `importScripts` keeps native synchronous ordering, repeated evaluation
and global declarations, including calls from ESM helpers. Module-worker calls
with ordinary string arguments (or no arguments) retain their native TypeError
without reading those script targets. Static GET fetches may use plain options;
their selectors and option effects remain. Other runtime networking is unchanged,
not proved closed. Native errors for unsupported host operations are not converted
into successful module loads.

Worker module cycles, recursive classic script imports and cyclic worker
construction still require the ordinary multi-file build. Closure identity,
nesting and finite-choice limits are enforced before publication. Realm-expanded
encoded worker resources also consume a cumulative byte budget, independently
of the final HTML size check. Open URL expressions, opaque host mutation,
location reflection and document classic-script Worker construction remain
outside this standalone route.

## Validation and boundaries

```sh
node --test tools/ingest/pack_module_graph.test.mjs \
  tools/ingest/pack_worker_realms.test.mjs \
  tools/ingest/pack_workers.test.mjs \
  tools/ingest/pack_classic_workers.test.mjs
```

The new module-graph suite compares original and emitted modules in separate
native Node worker threads. A resolve-only hook models the document import map;
Node owns ESM linking, JSON parsing, attributes, errors and module identity. The
existing production Blob factory is executed and its actual Blob is read for
worker-module tests. Positive emitted cases execute after their source directory
is renamed out of reach. No production source is substituted or rewritten by the
test harness.

The nested-worker suite executes original and emitted applications in real Node
threads, covering mixed classic/module descendants, repeated strict/sloppy classic
scripts, shared-helper bases, selected-only construction, transferable ArrayBuffer
ownership and native streaming Wasm. Its test-only host adapter supplies local
file responses, synchronous Script-goal importScripts, and browser-shaped message
events. Actual Blob source is read in its creating thread because Node Blob URLs
are thread-local. A raw-copy negative control fails after source removal.

Test teardown lets completed native modules exit or finish initial evaluation
before disposing their isolates. An earlier combined run exposed a Node/V8
SIGTRAP during forced termination at a result message; the original failure log
is retained rather than reported as a pass. No production behavior was changed
to avoid that test-host teardown race.

This is not browser HTML/Worker, HTTP/CORS, CSP, Trusted Types, SharedWorker or GPU
validation. The existing source-CSP refusal, root containment, fresh-output rule
and file/identity/depth/byte limits remain in force. No custom loader, evaluator,
fetch wrapper or worker runtime is added to the production artifact.
