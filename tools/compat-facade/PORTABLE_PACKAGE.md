# Portable retained-JavaScript compatibility package

This emitter packages an already prepared upstream runtime as an installable
`@franken/three` package. It is the retained-JavaScript compatibility lane from
plan sections 3.4, 4.7 and 5.12, not a compiled Rust/WASM engine or an acceleration
claim. It does not replace the existing development facade/import-map server.

## Generate and install

From the repository root, with the upstream runtime and build artifacts present:

```sh
node tools/compat-facade/portable-package.mjs \
  --package-dir upstream/three.js \
  --out dist/portable-three

cd dist/portable-three
npm pack --ignore-scripts --offline
```

The destination must not already exist and must be outside the source package.
The emitter does not build upstream, install dependencies, run lifecycle hooks,
or access the network. In a separate application, install the resulting tarball
with `npm install /absolute/path/to/franken-three-VERSION.tgz`.
Runtime dependencies are preserved; their installation may require registry
access or a populated npm cache. The integration tests use dependency-free
fixtures and run both packing and installation offline.

Use `--name three` to emit a package under the legacy name instead. These are
alternative package installations: independently installing both names creates
two physical module graphs, and cross-install singleton identity is not claimed.

The direct module API is also available:

```js
import { emitPortablePackage } from './tools/compat-facade/portable-package.mjs';

const { directory, manifest, files, count } = emitPortablePackage(
  'dist/portable-three',
  { packageDir: 'upstream/three.js', packageName: '@franken/three' },
);
```

## What is preserved

The runtime tree is copied to `_retained/`, with its original `package.json`.
The outer manifest redirects export targets into that tree, preserving ESM/CJS
conditions, conditional order, wildcard subpaths, fallback arrays and null
blockers. Internal `three` self-references resolve against the retained package,
so the root, WebGPU and addon paths share module identity when their upstream
targets do. CommonJS remains synchronous; no Promise-based `require` shim is used.

Relative imports, `import.meta.url` worker paths and binary assets retain their
layout. Declaration paths, versioned declarations, executable paths, local
browser mappings and side-effect glob paths are relocated. Runtime dependencies
and root license/notice files are retained.

Checkout metadata, installed `node_modules`, `.npmrc`, `.npmignore` and
`.gitignore` are not copied. Omitting nested ignore rules is deliberate: keeping
them can make `npm pack` silently discard a decoder or worker that exists in the
emitted folder. The original source is never rewritten. Source symlinks and
unsafe targets fail before publishing output; existing destinations are refused.
The tool expects a trusted, prepared runtime tree, not an arbitrary workspace
containing private files. Review its inventory before publishing externally.

## Validation

```sh
node --check tools/compat-facade/portable-package.mjs
node --test tools/compat-facade/portable-package.test.mjs \
  tools/compat-facade/portable-package.pack.test.mjs
```

The archive tests invoke real npm packing and offline installation for both
package names. They delete the source and emitted folder before installation,
check that every inventoried runtime file reached the archive, and exercise ESM,
synchronous CommonJS, addon identity, export blocking, a real Node worker, WASM
instantiation and a font URL relative to the worker. The fixture deliberately
contains upstream ignore rules that would otherwise remove those assets.

These tests do not establish full pinned-upstream compatibility, browser worker
behavior, bundler tree-shaking correctness, rendering parity or GPU performance.
