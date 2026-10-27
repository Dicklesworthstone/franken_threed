/** Opt-in library-kernel recognition in the real Rollup module graph. */
import fs from 'node:fs';
import {specializeMarchingCubesModule,specializeMarchingCubesBase} from './marching_cubes_specialization.mjs';
import {geometryNormalSourceRole,specializeGeometryNormals} from './geometry_normals_specialization.mjs';
import {specializeMatrixMultiply} from './matrix_multiply_specialization.mjs';
const ADAPTER='\0f3d-marching-cubes-adapter',RUNTIME='\0f3d-marching-cubes-runtime';
const NORMALS='\0f3d-geometry-normals-adapter',OBJECTS='\0f3d-numeric-objects';
const MATRICES='\0f3d-matrix-multiply-adapter';

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
  const reports=[],normalReports=[],matrixReports=[];let registrations=0,colorRegistrations=0;
  return {
    name:'f3d-marching-cubes',
    buildStart() {reports.length=0;normalReports.length=0;matrixReports.length=0;registrations=0;colorRegistrations=0;},
    resolveId(source,importer) {
      if ([ADAPTER,RUNTIME,NORMALS,OBJECTS,MATRICES].includes(source)) return source;
      if ([ADAPTER,NORMALS,MATRICES].includes(importer) && source==='./numeric_kernel_runtime.mjs') return RUNTIME;
      if ([ADAPTER,NORMALS].includes(importer) && source==='./numeric_objects.mjs') return OBJECTS;
      return null;
    },
    load(id) {
      if (id===ADAPTER) return fs.readFileSync(new URL('./marching_cubes_adapter.mjs',import.meta.url),'utf8');
      if (id===RUNTIME) return fs.readFileSync(new URL('./numeric_kernel_runtime.mjs',import.meta.url),'utf8');
      if (id===NORMALS) return fs.readFileSync(new URL('./geometry_normals_adapter.mjs',import.meta.url),'utf8');
      if (id===MATRICES) return fs.readFileSync(new URL('./matrix_multiply_adapter.mjs',import.meta.url),'utf8');
      if (id===OBJECTS) return fs.readFileSync(new URL('./numeric_objects.mjs',import.meta.url),'utf8');
      return null;
    },
    transform(code,id) {
      if ([ADAPTER,RUNTIME,NORMALS,OBJECTS,MATRICES].includes(id) || typeof code!=='string') return null;
      // Cheap discovery only. Complete source hashes, never this spelling or
      // the application's filename, authorize the transformation below.
      if (code.includes('class MarchingCubes')) {
        const result=specializeMarchingCubesModule(code,{runtimeModule:ADAPTER,maxMemoryPages,maxIterations});
        reports.push({id,...result.report});
        if (result.changed) return {code:result.code,map:null};
      }
      const edits=[],headers=[];let tail='';
      if (code.includes('class EventDispatcher') || code.includes('class Color')) {
        const base=specializeMarchingCubesBase(code,{runtimeModule:ADAPTER});
        if (base.changed) {
          if (base.registeredClasses.includes('EventDispatcher')) registrations++;
          if (base.registeredClasses.includes('Color')) colorRegistrations++;
          edits.push(...base.edits);headers.push(base.header);
        }
      }
      const normalCandidate=code.includes('class BufferGeometry') || code.includes('class BufferAttribute') || code.includes('class Vector3');
      const finish=(sources={})=>{
        if(normalCandidate) {
          const normals=specializeGeometryNormals(code,{...sources,runtimeModule:NORMALS,objectsModule:OBJECTS,maxMemoryPages,maxIterations});
          if(normals.report)normalReports.push({id,...normals.report});
          if(normals.changed){edits.push(...normals.edits);headers.push(normals.header);tail=normals.tail;}
        }
        if(code.includes('class Matrix4')) {
          const matrix=specializeMatrixMultiply(code,{runtimeModule:MATRICES,maxMemoryPages,maxIterations});
          if(matrix.report)matrixReports.push({id,...matrix.report});
          if(matrix.changed){edits.push(...matrix.edits);headers.push(matrix.header);tail+=matrix.tail;}
        }
        if(!headers.length)return null;
        let transformed=code;
        for(const edit of edits.sort((a,b)=>b.start-a.start))transformed=transformed.slice(0,edit.start)+edit.text+transformed.slice(edit.end);
        return {code:headers.join('')+transformed+tail,map:null};
      };
      if(normalCandidate && geometryNormalSourceRole(code)==='geometry') {
        return Promise.all([['vectorSource','../math/Vector3.js'],['attributeSource','./BufferAttribute.js']].map(async([key,specifier])=>{
          const resolved=await this.resolve(specifier,id,{skipSelf:true});
          if(resolved && !resolved.external)try {
            return [key,fs.readFileSync(resolved.id.startsWith('file:')?new URL(resolved.id):resolved.id,'utf8')];
          }catch{/* An unavailable source retains its normal module route. */}
          return [key,undefined];
        })).then(values=>finish(Object.fromEntries(values)));
      }
      return finish();
    },
    api:{getReport() {return {version:1,maxMemoryPages,maxIterations,compiledAddons:reports.filter(x=>x.route!=='retained-js').length,
      registeredBaseModules:registrations,registeredColorModules:colorRegistrations,
      compiledFieldKernels:reports.reduce((sum,report)=>sum+(report.compiledFieldKernels??0),0),accelerationClaim:false,modules:reports.map(x=>({...x}))};},
      getGeometryNormalsReport(){return {version:1,accelerationClaim:false,
        compiledModules:normalReports.filter(report=>report.compiledMethods.length).length,modules:normalReports.map(report=>({...report}))};},
      getMatrixMultiplicationReport(){return {version:1,accelerationClaim:false,
        compiledModules:matrixReports.length,modules:matrixReports.map(report=>({...report}))};}},
  };
}
