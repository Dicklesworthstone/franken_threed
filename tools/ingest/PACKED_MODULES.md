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

## Immutable lazy module selectors

Module ingestion, bundling and standalone packing follow lexical `const` string
bindings, including aliases captured by functions and finite conditional,
concatenation, template and sequence-result expressions:

```js
const mode = selectMode() ? 'viewer' : 'editor';
const entry = `./features/${mode}.mjs`;
export const open = () => import(entry);
```

Both modules are included while `selectMode()` and the chosen import still run
at their original boundaries. The source binding keeps its original value;
only the import argument receives a URL mapping. Native lazy evaluation, module
identity, live bindings, JSON attributes and temporal-dead-zone failures remain
host-owned. Document modules and packed module-worker helpers share this rule.

Parameters, mutable or imported bindings, destructured values and property/call
results are not assumed constant. Lexical shadowing is respected, including
function defaults and switch discriminants. Alias cycles and excessive expansion
remain unresolved: analysis admits at most 128 candidates, 64 nested expression
or alias steps and 64 KiB per expanded string; packing retains its 64-target
limit. Classic scripts keep their existing conservative import analysis.

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

## Retained Draco, KTX2 and Meshopt workers

Application builds retain finite glTF model and sidecar discovery through
`setDRACOLoader`, `setKTX2Loader` and `setMeshoptDecoder`. Authored finite Draco
decoder paths and Basis transcoder paths include their JavaScript and Wasm
resources at the original document-relative URLs. The pinned module-relative
defaults keep their existing `new URL(..., import.meta.url)` closure. Unproved
managers, escaping receivers and dynamic resource paths retain the host route.

With `prepareStandaloneLoaders`, the existing embedded resource resolver also
supports the retained Draco, KTX2 and Meshopt Blob workers. Exact pinned source-module
hashes identify their constructor sites before bundling. A small host function
forwards the original native Worker constructor and its arguments; the source
still creates the Blob payload, supplies the original Wasm bytes, schedules decoding,
maintains its worker pool and disposes workers and Blob URLs. The packer admits
that exact generated helper body. A matching filename or export name does not
admit a changed helper or arbitrary application Worker URL.

The Meshopt decoder keeps its embedded Wasm and source `useWorkers()` behavior.
Ordinary decoding and explicit worker pools share the same retained decoder;
worker reuse, transferred input/output buffers and shutdown stay source-owned.
No external Meshopt decoder-path mapping or alternate decoder is introduced.

Document-module Worker and URL checks use lexical bindings, so GLTFLoader's
unrelated local `URL` alias does not shadow a module-level codec asset URL.
Actual call-site shadows and global host writes still fail closure. Classic
worker realm checks retain their existing conservative behavior.

A Chromium offline run loaded real Draco-compressed geometry through both
retained JS and Wasm decoders and transcoded an ETC1S KTX2 texture. The packed
HTML ran after its source and emitted directories were made unavailable, using
three native Blob workers, zero application HTTP requests and no page errors.
This establishes that source decoder path; it does not replace the retained
loader or establish a decoder speedup.

An offline Meshopt check compared main-thread decoding and six real worker
decodes against the existing 48-byte golden output. The source GLTFLoader also
loaded an `EXT_meshopt_compression` / `KHR_mesh_quantization` model with four
points and identical backing bytes. Pool growth, reuse, shrinking and
`useWorkers(0)` shutdown created two native Blob workers in total; both closed,
and main-thread decoding still worked afterward. With source and build trees
unavailable, the packed application made no HTTP requests and reported no page
or console errors.

## Saved ObjectLoader scenes

Finite `ObjectLoader.load()` and `loadAsync()` requests now include external
images from the saved scene's `images` table, including cube-face URL arrays.
The model JSON remains byte-for-byte intact, and serialized data-image objects
and data URLs keep their source representation. The same closure copies image
files into normal application output and supplies the existing embedded loader
resource route when standalone preparation is enabled.

The source default image base is `setPath()` when nonempty, otherwise the first
load URL's directory. ObjectLoader retains that base on later loads until
`setResourcePath()` resets it. Shared loaders include all finite possible first
load bases and resource-path resets, so callbacks may execute in either order.
The path remains live while JSON is loading; a later reset also affects images
of an already pending scene. A dynamic base or reset,
custom manager, or escaping loader keeps its host-owned route; external URLs
are not downloaded. Explicit resource paths, encoded filenames and URL suffixes
retain their original logical identities. Invalid JSON is still delivered to
the source loader for its original parse failure.

The focused regression executes the retained source loader with external
requests rejected and source/build directories unavailable. It checks async and
callback loading, sticky image bases and pending resets, cube faces, embedded data textures,
independent instances, and editing/exporting/re-importing a loaded scene. Its
image host checks resource bytes and ownership, not browser image-decoder parity.

## Single-file addon assets

Finite document-relative loads through the pinned `HDRLoader`, `RGBELoader`,
`EXRLoader`, `FontLoader`, `STLLoader`, `PLYLoader` and `OBJLoader` include their
source files in application output and the prepared standalone resource map.
Named and namespace imports from their source modules or the addon aggregate
retain the same discovery rules. Decoder settings, including EXR part/format
selection and PLY property mappings, execute in the retained upstream loaders.
Original asset bytes, logical URL suffixes and callback ordering are preserved.

An executable packed regression decodes HDR and ZIP-compressed EXR pixels,
generates font outlines and loads STL/PLY/OBJ triangles after source/build files
are unavailable, with non-data requests rejected. This is retained-loader
packaging, not new renderer or browser GPU validation. Custom managers, escaping
loaders and unknown paths remain on their original host route; this does not
close material/texture sidecars for every model format.

## OBJ material libraries

Finite `MTLLoader.load()` and `loadAsync()` requests include their material
libraries and diffuse, specular, emissive, normal, bump, displacement and alpha
texture sidecars. `OBJLoader.setMaterials()` retains discovery of the OBJ model
itself, including fluent chains and separate setters. The original MTL and OBJ
bytes are unchanged; the retained source loaders still create materials, load
textures, parse geometry and preserve material-creator sharing or independent
instances.

Texture discovery follows the pinned MTL parser's repeated material/key rules,
first bump-map selection and `-bm`, `-mm`, `-s`, `-o` option handling. Unsupported
options remain part of the source URL instead of being interpreted by a new MTL
parser. A loader's default image base is captured when loading starts; finite
`setResourcePath()` changes also cover pending loads without applying
ObjectLoader's different sticky-base behavior. Encoded names and URL suffixes
retain their logical identities.

The exact pinned MTL module's read-only default-manager fallback does not count
as application exposure of that manager. This exception requires both its
canonical source path and full source hash. Modified or foreign modules and
actual application manager access retain the existing conservative route.

The packed regression runs the retained loaders after source and build files
are unavailable, with non-data requests rejected. It checks all seven texture
slots, UVs, texture transforms, material settings, async ordering, material
identity and a pending resource-path reset. Its image host verifies bytes and
loading completion; it does not establish browser decoder or renderer parity.
An independent Chrome 151 run of that packed artifact also passes with the
browser offline and every external request blocked: all 21 native PNG images
decode, all material/geometry assertions pass, and there are no browser errors.
This validates retained-loader packaging and native image decoding, not rendering
or acceleration.

## Validation and boundaries

```sh
node --test tools/ingest/dynamic_import_constants.test.mjs \
  tools/ingest/pack_module_graph.test.mjs \
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
