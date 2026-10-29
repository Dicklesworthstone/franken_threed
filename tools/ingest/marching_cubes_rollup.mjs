/** Opt-in library-kernel recognition in the real Rollup module graph. */
import fs from 'node:fs';
import {specializeMarchingCubesModule,specializeMarchingCubesBase} from './marching_cubes_specialization.mjs';
import {geometryNormalSourceRole,specializeGeometryNormals} from './geometry_normals_specialization.mjs';
import {specializeMatrixMultiply} from './matrix_multiply_specialization.mjs';
import {specializeTransformBatch} from './transform_batch_specialization.mjs';
import {specializeEulerUpdateBase,specializeEulerUpdateLoops} from './euler_update_specialization.mjs';
import {geometryTransformSourceRole,specializeGeometryTransforms} from './geometry_transforms_specialization.mjs';
import {specializeQuaternionFlat} from './quaternion_flat_specialization.mjs';
const ADAPTER='\0f3d-marching-cubes-adapter',RUNTIME='\0f3d-marching-cubes-runtime';
const NORMALS='\0f3d-geometry-normals-adapter',OBJECTS='\0f3d-numeric-objects';
const MATRICES='\0f3d-matrix-multiply-adapter',TRANSFORMS='\0f3d-transform-batch-adapter';
const ATTRIBUTE_TRANSFORMS='\0f3d-geometry-transforms-adapter';
const EULER_UPDATES='\0f3d-euler-update-adapter';
const QUATERNIONS='\0f3d-quaternion-flat-adapter';

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
  const reports=[],normalReports=[],matrixReports=[],transformReports=[],attributeReports=[],eulerReports=[],quaternionReports=[];let registrations=0,colorRegistrations=0;
  return {
    name:'f3d-marching-cubes',
    buildStart() {reports.length=0;normalReports.length=0;matrixReports.length=0;transformReports.length=0;attributeReports.length=0;eulerReports.length=0;quaternionReports.length=0;registrations=0;colorRegistrations=0;},
    resolveId(source,importer) {
      if ([ADAPTER,RUNTIME,NORMALS,OBJECTS,MATRICES,TRANSFORMS,ATTRIBUTE_TRANSFORMS,EULER_UPDATES,QUATERNIONS].includes(source)) return source;
      if ([ADAPTER,NORMALS,MATRICES,TRANSFORMS,ATTRIBUTE_TRANSFORMS,EULER_UPDATES,QUATERNIONS].includes(importer) && source==='./numeric_kernel_runtime.mjs') return RUNTIME;
      if ([TRANSFORMS,ATTRIBUTE_TRANSFORMS,EULER_UPDATES,QUATERNIONS].includes(importer) && source==='./matrix_multiply_adapter.mjs') return MATRICES;
      if (importer===EULER_UPDATES && source==='./transform_batch_adapter.mjs') return TRANSFORMS;
      if ([ADAPTER,NORMALS,ATTRIBUTE_TRANSFORMS].includes(importer) && source==='./numeric_objects.mjs') return OBJECTS;
      return null;
    },
    load(id) {
      if (id===ADAPTER) return fs.readFileSync(new URL('./marching_cubes_adapter.mjs',import.meta.url),'utf8');
      if (id===RUNTIME) return fs.readFileSync(new URL('./numeric_kernel_runtime.mjs',import.meta.url),'utf8');
      if (id===NORMALS) return fs.readFileSync(new URL('./geometry_normals_adapter.mjs',import.meta.url),'utf8');
      if (id===MATRICES) return fs.readFileSync(new URL('./matrix_multiply_adapter.mjs',import.meta.url),'utf8');
      if (id===TRANSFORMS) return fs.readFileSync(new URL('./transform_batch_adapter.mjs',import.meta.url),'utf8');
      if (id===ATTRIBUTE_TRANSFORMS) return fs.readFileSync(new URL('./geometry_transforms_adapter.mjs',import.meta.url),'utf8');
      if (id===EULER_UPDATES) return fs.readFileSync(new URL('./euler_update_adapter.mjs',import.meta.url),'utf8');
      if (id===QUATERNIONS) return fs.readFileSync(new URL('./quaternion_flat_adapter.mjs',import.meta.url),'utf8');
      if (id===OBJECTS) return fs.readFileSync(new URL('./numeric_objects.mjs',import.meta.url),'utf8');
      return null;
    },
    transform(code,id) {
      if ([ADAPTER,RUNTIME,NORMALS,OBJECTS,MATRICES,TRANSFORMS,ATTRIBUTE_TRANSFORMS,EULER_UPDATES,QUATERNIONS].includes(id) || typeof code!=='string') return null;
      // Cheap discovery only. Complete source hashes, never this spelling or
      // the application's filename, authorize the transformation below.
      if (code.includes('class MarchingCubes')) {
        const result=specializeMarchingCubesModule(code,{runtimeModule:ADAPTER,maxMemoryPages,maxIterations});
        reports.push({id,...result.report});
        if (result.changed) return {code:result.code,map:null};
      }
      const edits=[],headers=[];let tail='';
      if (code.includes('class EventDispatcher') || code.includes('class Color') ||
          code.includes('class Material') || code.includes('class ShaderMaterial')) {
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
        if(code.includes('class Quaternion')) {
          const quaternion=specializeQuaternionFlat(code,{runtimeModule:QUATERNIONS,maxMemoryPages,maxIterations});
          if(quaternion.report)quaternionReports.push({id,...quaternion.report});
          if(quaternion.changed){edits.push(...quaternion.edits);headers.push(quaternion.header);tail+=quaternion.tail;}
        }
        if(code.includes('class Object3D')) {
          const batch=specializeTransformBatch(code,{runtimeModule:TRANSFORMS,maxMemoryPages,maxIterations});
          if(batch.report)transformReports.push({id,...batch.report});
          if(batch.changed){edits.push(...batch.edits);headers.push(batch.header);}
          const euler=specializeEulerUpdateBase(code,{runtimeModule:EULER_UPDATES,maxMemoryPages,maxIterations});
          if(euler.report)eulerReports.push({id,...euler.report});
          if(euler.changed){edits.push(...euler.edits);headers.push(euler.header);}
        }
        if(normalCandidate || code.includes('class Matrix3')) {
          const attributes=specializeGeometryTransforms(code,{...sources,runtimeModule:ATTRIBUTE_TRANSFORMS,maxMemoryPages,maxIterations});
          if(attributes.report)attributeReports.push({id,...attributes.report});
          if(attributes.changed){edits.push(...attributes.edits);headers.push(attributes.header);tail+=attributes.tail;}
        }
        const eulerLoops=specializeEulerUpdateLoops(code,{runtimeModule:EULER_UPDATES});
        if(eulerLoops.report)eulerReports.push({id,...eulerLoops.report});
        if(eulerLoops.changed){edits.push(...eulerLoops.edits);headers.push(eulerLoops.header);}
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
      if(normalCandidate && geometryTransformSourceRole(code)==='attributes') {
        return this.resolve('../math/Vector3.js',id,{skipSelf:true}).then(resolved=>{
          let vectorSource;
          if(resolved && !resolved.external)try {
            vectorSource=fs.readFileSync(resolved.id.startsWith('file:')?new URL(resolved.id):resolved.id,'utf8');
          }catch{/* An unavailable source retains the attribute methods. */}
          return finish({vectorSource});
        });
      }
      return finish();
    },
    api:{getReport() {return {version:1,maxMemoryPages,maxIterations,compiledAddons:reports.filter(x=>x.route!=='retained-js').length,
      registeredBaseModules:registrations,registeredColorModules:colorRegistrations,
      compiledFieldKernels:reports.reduce((sum,report)=>sum+(report.compiledFieldKernels??0),0),accelerationClaim:false,modules:reports.map(x=>({...x}))};},
      getGeometryNormalsReport(){return {version:1,accelerationClaim:false,
        compiledModules:normalReports.filter(report=>report.compiledMethods.length).length,modules:normalReports.map(report=>({...report}))};},
      getMatrixMultiplicationReport(){return {version:1,accelerationClaim:false,
        compiledModules:matrixReports.length,modules:matrixReports.map(report=>({...report}))};},
      getQuaternionFlatReport(){return {version:1,accelerationClaim:false,
        compiledModules:quaternionReports.length,modules:quaternionReports.map(report=>({...report}))};},
      getTransformBatchesReport(){return {version:1,accelerationClaim:false,
        compiledModules:transformReports.length,modules:transformReports.map(report=>({...report}))};},
      getEulerUpdatesReport(){return {version:1,accelerationClaim:false,
        compiledModules:eulerReports.filter(report=>report.kind==='pinned-euler-kernel').length,
        compiledLoops:eulerReports.reduce((sum,report)=>sum+(report.compiledLoops??0),0),modules:eulerReports.map(report=>({...report}))};},
      getGeometryTransformsReport(){return {version:1,accelerationClaim:false,
        compiledModules:attributeReports.filter(report=>report.compiledMethods.length).length,modules:attributeReports.map(report=>({...report}))};}},
  };
}
