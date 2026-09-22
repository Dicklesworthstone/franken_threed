/** Opt-in library-kernel recognition in the real Rollup module graph. */
import fs from 'node:fs';
import {specializeMarchingCubesModule,specializeMarchingCubesBase} from './marching_cubes_specialization.mjs';
const ADAPTER='\0f3d-marching-cubes-adapter',RUNTIME='\0f3d-marching-cubes-runtime';

/**
 * Place before generic URL resolvers so these private virtual IDs never become
 * user asset paths. Only source-pinned addon/base modules are transformed. All
 * runtime imports are ordinary modules, so Rollup performs lexical deconfliction
 * before any later numeric chunk specialization. No browser parser/compiler ships.
 */
export function marchingCubesRollupPlugin(options={}) {
  if (!options || typeof options!=='object' || Array.isArray(options)) throw new TypeError('Expected marching-cubes options');
  for (const key of Object.keys(options)) {
    if (!['maxMemoryPages','maxIterations'].includes(key)) throw new TypeError(`Unknown marching-cubes option: ${key}`);
  }
  const {maxMemoryPages=2048,maxIterations=100000000}=options;
  if (!Number.isInteger(maxMemoryPages) || maxMemoryPages<1 || maxMemoryPages>16384 ||
      !Number.isInteger(maxIterations) || maxIterations<1 || maxIterations>1000000000) {
    throw new RangeError('Invalid marching-cubes memory or iteration budget');
  }
  const reports=[];let registrations=0,colorRegistrations=0;
  return {
    name:'f3d-marching-cubes',
    buildStart() {reports.length=0;registrations=0;colorRegistrations=0;},
    resolveId(source,importer) {
      if (source===ADAPTER || source===RUNTIME) return source;
      if (importer===ADAPTER && source==='./numeric_kernel_runtime.mjs') return RUNTIME;
      return null;
    },
    load(id) {
      if (id===ADAPTER) return fs.readFileSync(new URL('./marching_cubes_adapter.mjs',import.meta.url),'utf8');
      if (id===RUNTIME) return fs.readFileSync(new URL('./numeric_kernel_runtime.mjs',import.meta.url),'utf8');
      return null;
    },
    transform(code,id) {
      if (id===ADAPTER || id===RUNTIME || typeof code!=='string') return null;
      // Cheap discovery only. Complete source hashes, never this spelling or
      // the application's filename, authorize the transformation below.
      if (code.includes('class MarchingCubes')) {
        const result=specializeMarchingCubesModule(code,{runtimeModule:ADAPTER,maxMemoryPages,maxIterations});
        reports.push({id,...result.report});
        if (result.changed) return {code:result.code,map:null};
      }
      if (code.includes('class EventDispatcher') || code.includes('class Color')) {
        const result=specializeMarchingCubesBase(code,{runtimeModule:ADAPTER});
        if (result.changed) {
          if (result.registeredClasses.includes('EventDispatcher')) registrations++;
          if (result.registeredClasses.includes('Color')) colorRegistrations++;
          return {code:result.code,map:null};
        }
      }
      return null;
    },
    api:{getReport() {return {version:1,maxMemoryPages,maxIterations,compiledAddons:reports.filter(x=>x.route!=='retained-js').length,
      registeredBaseModules:registrations,registeredColorModules:colorRegistrations,
      compiledFieldKernels:reports.reduce((sum,report)=>sum+(report.compiledFieldKernels??0),0),accelerationClaim:false,modules:reports.map(x=>({...x}))};}},
  };
}
