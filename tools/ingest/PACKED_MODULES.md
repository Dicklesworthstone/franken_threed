# Native modules in single-HTML exports

`packHtml` closes the static module graph of an already emitted application.
The ordinary multi-file build is not modified. This is retained-JavaScript
packaging (F3D-10), not a renderer, compiler specialization or speedup claim.

## Project transforms before bundling

The Node build API accepts the project's existing synchronous TypeScript, JSX
or other JavaScript frontend before module analysis and Rollup linking:

```js
await buildApplication('./src/main.ts', './dist', {
  transformModule(source, { url, sourceType }) {
    // frontend is the application's already configured transformer.
    return frontend.transformSync(source, { filename: fileURLToPath(url), sourceType });
  },
  specializeNumeric: true,
  routeWebGPURenderer: true,
});
```

Return a JavaScript string, `{ code }`, or `null`/`undefined` to retain the input.
`buildModuleGraph`, `bundleWithRollup` and worker-graph APIs accept the same
option. A build caches one normalized source per exact URL and module/script
kind; analysis, source hashes, renderer decisions, retained file copies and
emission use that same snapshot. Query and fragment identities remain distinct.
Inline module entries, finite lazy dependencies, module workers and classic
worker `importScripts` dependencies participate. External classic scripts also
use their normalized bytes; inline classic HTML scripts remain source-owned.
Unchanged retained scripts and JSON/binary resources preserve their bytes.

Installed and linked workspace packages can export `.ts`, `.tsx`, `.mts`,
`.cts` or `.jsx` sources through their normal `exports`, `imports` and browser
maps. The selected canonical URL reaches the same cached frontend before the
package format check. Return JavaScript explicitly to handle that source;
returning `null`/`undefined` leaves an unsupported package format rejected.
An explicit unchanged string is accepted, so a type-free source module does not
need a cosmetic edit. The hook must preserve the intended module semantics;
CommonJS is not automatically converted. JSON and binary package targets do not
enter the JavaScript frontend, and package export selection is unchanged.

The CLI loads an explicitly selected local frontend module:

```sh
node tools/ingest/cli.mjs --entry ./src/main.ts --build-app ./dist \
  --transform-module ./frontend.mjs
```

That module must have a named `transformModule` function export with the same
signature and return contract as the Node API. It may await frontend setup at
module scope; each transform call must remain synchronous. The option also
works for graph-only ingestion and with `--build-app --pack-html`. Standalone
packing without a build, animation export and individual kernel compilation do
not consume this frontend option.

Imports and assets keep their original filesystem URL bases. Frontend-generated
assets must exist at the referenced paths, and imports must remain resolvable
through the existing file/package/import-map resolver. This handoff does not
implement Vite virtual modules, arbitrary plugin resolution, a TypeScript
compiler, or a source-map merger. Diagnostics and graph spans describe normalized
JavaScript. An asynchronous frontend can complete its work first and return
cached code through this synchronous hook; returning a Promise fails before
output publication.

Native renderer routing requires unchanged Three.js library input. A frontend
edit to an admitted build/core module preserves that actual transformed library
and prevents native substitution. Exact loader and numeric-library recognition
continues to use source hashes. Unchanged pinned imports retain their original
module identities, so typed applications can select the existing native renderer
and numeric Wasm routes before bundling removes those boundaries.

The focused integration executes real TypeScript normalization, numeric Wasm
calls, lazy/live bindings, assets, module workers and repeated classic imports.
Its packed application also runs with both source and emitted directories
unavailable. Separate checks construct the native renderer from typed source and
execute a modified pinned core through the retained route. These establish input
integration and execution ownership; they do not establish a rendering speedup.

## Native document fetch resources

HTML application builds include finite local GET resources requested by document
modules and retained classic scripts. Plain strings, immutable module `const`
aliases, conditional choices, string concatenation and template selectors use
the existing bounded expression analysis. Resources resolve from the document's
effective base, including supported local `<base>` paths in the ordinary build,
regardless of which imported module contains the call. Normalized project source
participates before bundling. Missing files, remote URLs and unproved requests
keep their original networked routes.

Standalone packing maps admitted URL arguments to embedded resources and keeps
the original native `fetch` call. Selector bindings, query/fragment strings,
evaluation order, receiver, plain `RequestInit` expressions, abort signals,
integrity checks and response-body consumption remain source-owned. There is no
global fetch patch, synthetic Response or runtime request handler. Fetched
JavaScript, CSS, SVG and model files are raw response bytes: they are not executed,
passed through the project frontend, or recursively rewritten as active assets.
Wasm retains the exact `application/wasm` MIME needed by native streaming APIs.

Locally bound or imported functions named `fetch` are not treated as the browser
API. Known host replacement, non-GET methods and opaque options prevent embedding
finite requests. `Request` objects, fetch aliases and runtime-selected URLs keep
the existing host path; their closure is not newly proved. Remote and document-root
requests still need ordinary deployment, and the packer's existing `<base>`
refusal remains. Embedded Response URLs are data URLs; server HTTP headers, cache
behavior and live endpoints are not reproduced by this static-resource mode.

The focused execution compares source and emitted applications served over HTTP,
then closes both servers and runs the packed application with its source and build
trees unavailable. Native fetch checks raw bodies, cloned and streamed responses,
finite URL choices, abort reasons, invalid GET bodies and failed integrity, plus
audio bytes and `WebAssembly.instantiateStreaming()`. This proves resource
packaging and retained host execution, not rendering or acceleration.

An independent Chrome 151.0.7922.34 run opens that packed HTML with the browser
offline, both original trees unavailable, and no fetch or response interception.
All response/body and streaming-Wasm assertions pass with zero external requests,
page errors or unexpected console errors. The intentionally invalid integrity
request produces two expected console diagnostics and its caught native failure.
Native `AudioContext.decodeAudioData()` decodes the embedded 52-byte PCM WAV into
one channel at 8,000 Hz with four frames.

## Typed imports and source errors

Static imports, reexports and bounded lazy imports retain their original import
attributes. Document and worker module identities include the requested type as
well as the source URL, including query/fragment variants. JSON bytes are copied
unchanged: no object-literal wrapper, eager JSON parsing, BOM removal or numeric
round trip occurs during packing. An unselected malformed JSON dependency does
not fail the build; the native importer reports its error when selected.

Installed and linked workspace packages may select JSON through `exports`,
private `imports` and browser mappings. Those selected resources use the same
native JSON path as relative imports in application builds and module workers;
the project JavaScript frontend never receives their bytes. Missing attributes
do not relabel JSON as JavaScript, and requesting JSON does not relabel a
JavaScript resource. Static reexports and lazy imports share the selected URL's
instance, while query/fragment variants and separate files remain distinct.

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

## 3MF manufacturing models

Finite `ThreeMFLoader.load()` and `loadAsync()` calls now include their complete
3MF archives in application output and the prepared standalone resource map.
Source discovery recognizes the actual `3MFLoader.js` module filename as well
as named and namespace addon-aggregate imports. Constructor aliases, finite URL
choices, `setPath`, query/fragment suffixes and ordinary `parse()` usage retain
the existing loader discovery rules. Custom extensions, custom managers and
escaping loaders keep their original host route.

The ZIP bytes are copied unchanged. The retained Three.js loader and fflate
decode the original model parts, materials and component hierarchy at runtime;
the source loader creates and retires its own image Blob URLs. Packaging does
not extract the archive, rewrite its XML, or substitute a model parser.

Three focused cases and the complete loader-asset suite pass (34/34).
`three_mf_package.browser.mjs` compares the original HTTP-loaded source models
with offline packed execution after source/build directories become unavailable.
The upstream truck fixture retains 40 meshes, 615,930 vertices, 205,310 triangles,
both component trees, transforms, material values, matching attribute/index
hashes and decoded pixels from its 1352-by-339 PNG. Repeated face-color model loads remain
independent, with source asynchronous callback ordering. Chromium 151.0.7922.34
reports zero external requests or browser errors. This proves retained-loader
packaging and browser image decoding; it does not claim new GPU execution or a
decoder speedup.

## Rhino3DM CAD models and runtime decoding

Finite `Rhino3dmLoader` requests from the pinned `3DMLoader.js` module or addon
aggregate include the original 3DM file and both runtime dependencies,
`rhino3dm.js` and `rhino3dm.wasm`. `setLibraryPath()` retains its literal prefix
semantics; the default empty path resolves from the document. Finite library
changes while a model request is pending are included because the retained loader
initializes its decoder after the model response arrives. `setPath()`, worker
limits and subdivision settings keep their source behavior.

The codec is also included for `parse()`, `parseAsync()` and `decodeObjects()`
with runtime bytes, so an offline application can open models that were not known
when it was built. Model and decoder bytes are unchanged. The source loader and
the application's actual rhino3dm library own geometry, curves, SubD processing,
instance definitions, material/image decoding, document metadata and errors;
this route does not preconvert the model or replace the CAD engine.

With `prepareStandaloneLoaders`, the exact pinned loader's Blob-worker constructor
uses the existing native worker forwarding boundary. Worker payload creation,
Wasm initialization, transferable buffers, task caching, worker pools and disposal
remain source-owned. Modified or foreign loader source receives no worker hook.
Custom managers, escaped receivers and unproved library paths retain their host
routes; a remote library path still requires the corresponding network deployment.

The focused browser comparison uses the source-recommended rhino3dm 8.32.1,
verified against its original JS and Wasm hashes. It compares the upstream Rhino
logo's CAD geometry, materials, layers and document metadata with packed offline
execution after the source and build directories become unavailable. A second
model, created after packing, exercises runtime mesh/curve parsing, an embedded
PNG, independent decoders, buffer transfer, malformed input and worker recreation
after disposal. This is retained CAD/runtime functionality, not Rust ownership or
renderer acceleration evidence.

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

## Collada scenes, textures and animation

Pinned `ColladaLoader` imports, aliases and addon-aggregate imports now participate
in finite standalone resource closure. The package preserves original DAE, image
and TGA bytes, the path captured when loading starts, and resource-path changes
while a model is pending. Image URLs retain the source's literal path-prefix
concatenation, query/fragment identity and XML text semantics.

A bounded metadata reader discovers the first direct image library and follows
the pinned duplicate-ID and `init_from` rules. The original XML parser, geometry,
skeletons, animation, kinematics, texture decoding and runtime `parse()` remain
source-owned. Malformed documents retain their native error behavior. Valid DTD
or custom-entity metadata is explicitly refused with `COLLADA_XML_UNSUPPORTED`
when closing a standalone package; it is not silently classified as a closed
resource graph or automatically redirected to another deployment mode. Unknown
paths, custom managers and escaped loaders retain their existing host boundaries.

Five focused loader cases cover discovery, metadata, texture bases, an actual
animated-model build and the valid-DTD refusal; all 38 adjacent loader checks
also pass. `node tools/ingest/collada_package.browser.mjs` compares the pinned
source with a packed page after the source/build directories and server become
unavailable. Chromium 151.0.7922.34 matches the skinned stormtrooper's 19,554
vertices, 6,518 triangles and three animated poses, independent reloads, the
42,624-vertex elf and four decoded 1024-by-1024 JPEG images, pending path resets,
callbacks, runtime TGA parsing and malformed XML. There are no external requests
or unexpected browser errors. The source's unsupported morph-animation warning
is preserved; this is retained loader functionality, without a performance or
complete loader-family claim.

## FBX scenes, textures and animation

Finite pinned `FBXLoader.load()` and `loadAsync()` requests now package their
original model bytes and connected external image files. ASCII FBX from version
7000 and binary FBX from version 6400 are inspected only for resource metadata,
including the 64-bit node headers introduced in version 7500. Geometry arrays,
compressed payloads, skeletal bindings, morph data and animations stay with the
retained source parser. Boolean `trimAnimationClips` assignments remain live
source settings; unknown mutations or replacing that field with a callable keep
the opaque host route.

Image discovery follows each Texture node's first connection to a Video node,
uses `RelativeFilename` before `Filename`, and applies the source's Windows-path
basename rule. Unused Videos and later connection children do not add resources.
Embedded image content stays embedded, including the source's shared-filename
deduplication across Video nodes. The metadata scanner does not use a Texture's
`FileName` as its image URL or decompress geometry to discover dependencies.

As with MTL, each load captures its default path while a finite resource-path
reset can affect an already pending parse. FBX uses the literal texture-loader
prefix, including for an authored HTTP URL; it does not inherit MTL's different
absolute-URL exception. Custom managers, unknown paths and escaping loaders
retain their source host behavior. Metadata inspection has explicit byte,
node, nesting, property-count and offset bounds. Malformed metadata leaves the
original model bytes intact for the source parse error; resource-budget failures
remain explicit.

The packed execution regression loads real pinned binary geometry and animation
plus an ASCII skinned model after source and build files are unavailable. It
checks texture completion, independent model instances, a pending resource-path
reset, a quaternion animation sample, skeletal bone motion and trimmed clips,
with non-data requests rejected. A complementary regression preserves ASCII,
compressed-array binary and embedded-image binary triangle models, checking
UVs, material color space, independent loader results and native Blob-image
loading. Both suites retain the original loader and model bytes.

Independent Chrome 151.0.7922.34 offline runs cover both applications. The
authored triangle case decodes five native PNG images, including the source
Blob image, and checks exact red RGBA bytes. The real-model case decodes three
1024-by-1024 PNG images and preserves Suzanne's 2,910 vertices, the animated
cube, and the skinned ball's two trimmed clips and animated bone sample. Both
runs report zero external requests and browser errors. This establishes
retained-loader packaging and native image decoding; rendering and acceleration
are separate claims.

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
