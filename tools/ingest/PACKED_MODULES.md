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

## Validation and boundaries

```sh
node --test tools/ingest/pack_module_graph.test.mjs \
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

This is not browser HTML/Worker, HTTP/CORS, CSP, Trusted Types, SharedWorker or GPU
validation. The existing source-CSP refusal, root containment, fresh-output rule
and file/identity/depth/byte limits remain in force. No custom loader, evaluator,
fetch wrapper or worker runtime is added to the production artifact.
