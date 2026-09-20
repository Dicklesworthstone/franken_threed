/**
 * Relocatable retained-upstream packages (Plan sections 3.4, 4.7 and 5.12).
 * The retained package keeps its own package.json: relative imports, package
 * self-references, workers and decoder URLs therefore keep their original base.
 * This is compatibility execution, not a Rust/GPU acceleration claim.
 */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const RETAINED = '_retained';
// Checkout and package-manager control files are not runtime assets. In
// particular, nested ignore files can silently strip decoder WASM from npm pack.
const OMIT_NAMES = new Set(['.git', 'node_modules', '.gitignore', '.npmignore', '.npmrc']);
const PACKAGE_NAME = /^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/;

function exists(file) {
  try { fs.lstatSync(file); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

function within(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`));
}

// Resolve existing ancestors as well: a symlinked output parent must not sneak
// the output into the source tree and recursively include its own package.
function canonicalDestination(destination) {
  let ancestor = path.resolve(destination);
  const missing = [];
  while (!exists(ancestor)) {
    missing.unshift(path.basename(ancestor));
    ancestor = path.dirname(ancestor);
  }
  return path.join(fs.realpathSync(ancestor), ...missing);
}

function retainedTarget(value, { external = false, relative = false } = {}) {
  if (typeof value !== 'string') throw new TypeError('Package targets must be strings');
  if (external && !value.startsWith('.')) return value;
  const target = relative && !value.startsWith('./') ? `./${value}` : value;
  if (!target.startsWith('./') || /[\\\0?#]/.test(target)) {
    throw new Error(`Unsupported or escaping package target: ${value}`);
  }
  let decoded;
  try { decoded = decodeURIComponent(target.slice(2)); }
  catch { throw new Error(`Invalid encoded package target: ${value}`); }
  if (!decoded || /[\\\0]/.test(decoded) || decoded.split('/').some((part) => part === '..' || part === '.' || part === 'node_modules' || part === '')) {
    throw new Error(`Unsupported or escaping package target: ${value}`);
  }
  return `./${RETAINED}/${target.slice(2)}`;
}

function mapTargets(value, transform) {
  if (value === null || value === false) return value;
  if (typeof value === 'string') return transform(value);
  if (Array.isArray(value)) return value.map((target) => mapTargets(target, transform));
  if (typeof value !== 'object') throw new TypeError('Invalid conditional package target');
  return Object.fromEntries(Object.entries(value).map(([key, target]) => [key, mapTargets(target, transform)]));
}

/** Preserve conditional ordering, arrays, wildcard patterns and null blockers. */
export function portablePackageManifest(upstream, { packageName = '@franken/three' } = {}) {
  if (!PACKAGE_NAME.test(packageName)) throw new Error(`Invalid package name: ${packageName}`);
  if (!upstream || typeof upstream !== 'object' || Array.isArray(upstream) || !Object.hasOwn(upstream, 'exports')) {
    throw new Error('Portable packages require an explicit upstream exports contract');
  }
  const manifest = { ...upstream, name: packageName };
  manifest.exports = mapTargets(upstream.exports, (target) => retainedTarget(target));
  for (const key of ['main', 'module', 'types', 'typings']) {
    if (upstream[key] !== undefined) manifest[key] = retainedTarget(upstream[key], { relative: true });
  }
  if (upstream.imports) manifest.imports = mapTargets(upstream.imports, (target) => retainedTarget(target, { external: true }));
  if (typeof upstream.browser === 'string') manifest.browser = retainedTarget(upstream.browser, { relative: true });
  else if (upstream.browser && typeof upstream.browser === 'object') {
    manifest.browser = Object.fromEntries(Object.entries(upstream.browser).map(([key, value]) => [
      key.startsWith('.') ? retainedTarget(key) : key,
      typeof value === 'string' ? retainedTarget(value, { external: true }) : value,
    ]));
  }
  if (Array.isArray(upstream.sideEffects)) {
    manifest.sideEffects = upstream.sideEffects.map((pattern) => {
      if (typeof pattern !== 'string' || pattern.startsWith('!')) {
        throw new TypeError('Unsupported sideEffects pattern');
      }
      // A basename-only pattern applies at every depth, not just at the root.
      return retainedTarget(pattern.includes('/') ? pattern : `**/${pattern}`, { relative: true });
    });
  } else if (upstream.sideEffects !== undefined && typeof upstream.sideEffects !== 'boolean') {
    throw new TypeError('sideEffects must be a boolean or an array');
  }
  if (upstream.typesVersions) {
    manifest.typesVersions = mapTargets(upstream.typesVersions, (target) => retainedTarget(target, { relative: true }));
  }
  if (typeof upstream.bin === 'string') manifest.bin = retainedTarget(upstream.bin, { relative: true });
  else if (upstream.bin) manifest.bin = Object.fromEntries(Object.entries(upstream.bin).map(([key, value]) => [key, retainedTarget(value, { relative: true })]));
  // Upstream development/publish hooks must not run when installing this facade.
  for (const key of ['scripts', 'devDependencies', 'workspaces', 'directories', 'publishConfig']) delete manifest[key];
  manifest.files = [RETAINED];
  return manifest;
}

function packageFiles(root, directory = '') {
  const files = [];
  for (const name of fs.readdirSync(path.join(root, directory)).sort()) {
    if (OMIT_NAMES.has(name)) continue;
    const relative = path.join(directory, name);
    const stat = fs.lstatSync(path.join(root, relative));
    if (stat.isSymbolicLink()) throw new Error(`Package symlinks are not portable: ${relative}`);
    if (stat.isDirectory()) files.push(...packageFiles(root, relative));
    else if (stat.isFile()) files.push(relative);
    else throw new Error(`Unsupported package file: ${relative}`);
  }
  return files;
}

/**
 * Emit an installable package with no filesystem references to the source tree.
 * Runtime files are retained, including files not exported as JS modules.
 * Package-manager ignore/config files and checkout metadata are not copied.
 * Existing destinations are never intentionally replaced. The source is not
 * modified and no dependency install, lifecycle script or network request runs.
 *
 * @param {string} targetDir New output directory.
 * @param {{packageDir?: string, packageName?: string}} options
 * @returns {{directory: string, manifest: object, files: string[], count: number}}
 */
export function emitPortablePackage(targetDir, options = {}) {
  if (typeof targetDir !== 'string' || !targetDir) throw new TypeError('A destination directory is required');
  const source = fs.realpathSync(options.packageDir || 'upstream/three.js');
  const destination = canonicalDestination(targetDir);
  if (exists(destination)) throw new Error(`Destination already exists: ${destination}`);
  if (within(source, destination)) throw new Error('Portable output must be outside the source package');
  const files = packageFiles(source);
  const upstream = JSON.parse(fs.readFileSync(path.join(source, 'package.json'), 'utf8'));
  const manifest = portablePackageManifest(upstream, options);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  const staging = fs.mkdtempSync(path.join(path.dirname(destination), '.f3d-package-'));
  try {
    for (const relative of files) {
      const from = path.join(source, relative);
      const to = path.join(staging, RETAINED, relative);
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.copyFileSync(from, to);
      fs.chmodSync(to, fs.statSync(from).mode & 0o777);
    }
    // Keep license/notice files visible to package registries as well as in the
    // unchanged retained package. No source headers or licenses are removed.
    for (const relative of files.filter((file) => !file.includes(path.sep) && /^(?:licen[cs]e|copying|notice)(?:[.-]|$)/i.test(file))) {
      fs.copyFileSync(path.join(source, relative), path.join(staging, relative));
      manifest.files.push(relative);
    }
    fs.writeFileSync(path.join(staging, 'package.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    if (exists(destination)) throw new Error(`Destination already exists: ${destination}`);
    fs.renameSync(staging, destination);
  } catch (error) {
    fs.rmSync(staging, { recursive: true, force: true });
    throw error;
  }
  return { directory: destination, manifest, files: files.map((file) => `${RETAINED}/${file.split(path.sep).join('/')}`), count: files.length };
}

function main(args) {
  const options = {};
  let out;
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (!['--out', '--package-dir', '--name'].includes(flag) || !args[i + 1] || args[i + 1].startsWith('--')) {
      throw new Error('Usage: node portable-package.mjs --out NEW_DIRECTORY [--package-dir UPSTREAM] [--name @franken/three]');
    }
    const value = args[++i];
    if (flag === '--out') out = value;
    else if (flag === '--package-dir') options.packageDir = value;
    else options.packageName = value;
  }
  const result = emitPortablePackage(out, options);
  console.log(JSON.stringify({ directory: result.directory, retainedFiles: result.count, execution: 'retained-upstream-js', accelerated: false }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { main(process.argv.slice(2)); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
