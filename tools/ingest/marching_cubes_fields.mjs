/** Build-time lifting of the pinned addon's field loops into the shared compiler.
 * Retain color conversion, scalar setup and blur's slice in the original method.
 * Only effect-free owner/color data loads are replaced by guarded ABI values.
 * No alternate metaball formula, reassociation or synthetic lookup table.
 */
import {createHash} from 'node:crypto';
import * as acorn from 'acorn';
import {compileNumericKernel} from './numeric_kernel.mjs';

const SOURCE_BLOB='29a405be3eae30a7e2b1ff04827068921d31d5dc';
const scalar=name=>({name,type:'f64',kind:'local',property:name});
const localArray=name=>({name,type:'f32[]',kind:'local',property:name});
const owner=(property,type='f64')=>({name:'__f3d_'+property,type,kind:'owner',property});
const color=property=>({name:'__f3d_color_'+property,type:'f64',kind:'color',property});
const planes=[localArray('field'),...['size','yd','zd','strength','subtract','dist'].map(scalar)];
const recipes={
  addBall:[owner('field','f32[]'),owner('palette','f32[]'),owner('size'),owner('size2'),
    ...['ballx','bally','ballz','strength','subtract','sign','radius','zs','ys','xs',
      'min_z','max_z','min_y','max_y','min_x','max_x'].map(scalar),...['r','g','b'].map(color)],
  addPlaneX:planes,addPlaneY:planes,addPlaneZ:planes,
  blur:[localArray('field'),localArray('fieldCopy'),...['size','size2','intensity'].map(scalar)],
  reset:[owner('normal_cache','f32[]'),owner('field','f32[]'),owner('palette','f32[]'),owner('size3')],
};

function visit(node,callback) {
  if (!node || typeof node.type!=='string') return;
  if (callback(node)===false) return;
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) for (const child of value) visit(child,callback);
    else if (value && typeof value==='object') visit(value,callback);
  }
}
const member=(node,object,property)=>node?.type==='MemberExpression' && !node.computed &&
  !node.optional && (object==='this' ? node.object.type==='ThisExpression'
    : node.object.type==='Identifier' && node.object.name===object) && node.property.name===property;

/** Source must be the independently pinned complete addon, not a caller hash. */
export function compileMarchingCubesFields(source,{maxMemoryPages=2048,maxIterations=100000000}={}) {
  if (typeof source!=='string') throw new TypeError('Expected pinned MarchingCubes source');
  const bytes=Buffer.from(source);
  const hash=createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
  if (hash!==SOURCE_BLOB) throw new TypeError('MarchingCubes field source pin mismatch');
  const ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module'});
  const cls=ast.body.find(node=>node.type==='ClassDeclaration' && node.id.name==='MarchingCubes');
  const ctor=cls?.body.body.find(node=>node.kind==='constructor')?.value;
  if (!ctor) throw new TypeError('Missing pinned constructor');
  const result=[];
  for (const [name,parameters] of Object.entries(recipes)) {
    const assignment=ctor.body.body.find(node=>node.type==='ExpressionStatement' &&
      node.expression.type==='AssignmentExpression' && member(node.expression.left,'this',name))?.expression;
    const fn=assignment?.right;
    const loops=fn?.body.body.filter(node=>node.type==='ForStatement');
    if (fn?.type!=='FunctionExpression' || loops?.length!==1 || fn.body.body.at(-1)!==loops[0]) {
      throw new TypeError(`Missing pinned ${name} loop boundary`);
    }
    const loop=loops[0],edits=[];
    const rename=text=>{
      const parsed=acorn.parse(`(${text})`,{ecmaVersion:'latest'}),changes=[];
      visit(parsed,node=>{
        for (const parameter of parameters) {
          if (parameter.kind==='local') continue;
          if (member(node,parameter.kind==='owner'?'this':'ballColor',parameter.property)) {
            changes.push({start:node.start-1,end:node.end-1,text:parameter.name});return false;
          }
        }
      });
      for (const change of changes.sort((a,b)=>b.start-a.start)) text=text.slice(0,change.start)+change.text+text.slice(change.end);
      return text;
    };
    visit(loop,node=>{
      // reset's chained zero stores have no observable reference evaluation
      // after descriptor/native-view guards. Preserve their right-to-left stores.
      if (node.type==='ExpressionStatement' && node.expression.type==='AssignmentExpression' &&
          node.expression.right.type==='AssignmentExpression') {
        const targets=[];let expr=node.expression;
        while (expr.type==='AssignmentExpression') {
          if (expr.operator!=='=' || expr.left.type!=='MemberExpression') throw new TypeError('Unexpected pinned assignment chain');
          targets.push(rename(source.slice(expr.left.start,expr.left.end)));expr=expr.right;
        }
        if (expr.type!=='Literal' || expr.value!==0) throw new TypeError('Only pinned zero-store chains are lifted');
        edits.push({start:node.start,end:node.end,text:targets.reverse().map(target=>`${target}=0;`).join('\n')});
        return false;
      }
      for (const parameter of parameters) {
        if (parameter.kind!=='local' && member(node,parameter.kind==='owner'?'this':'ballColor',parameter.property)) {
          edits.push({start:node.start,end:node.end,text:parameter.name});return false;
        }
      }
    });
    let body=source.slice(loop.start,loop.end);
    for (const edit of edits.sort((a,b)=>b.start-a.start)) {
      body=body.slice(0,edit.start-loop.start)+edit.text+body.slice(edit.end-loop.start);
    }
    // The pinned loops assign all these numeric scratch locals before reading
    // them. Their uninitialized JS values do not escape the method or loop.
    const scratch=fn.body.body.filter(node=>node.type==='VariableDeclaration' && node.start<loop.start)
      .flatMap(node=>node.declarations.filter(item=>item.init===null).map(item=>item.id.name));
    const kernelSource=`function ${name}(${parameters.map(p=>p.name).join(',')}) {\n`+
      (scratch.length?`let ${scratch.map(local=>local+'=0').join(',')};\n`:'')+body+'\n}';
    const artifact=compileNumericKernel(kernelSource,{parameterTypes:parameters.map(p=>p.type),
      sourceName:`f3d:marching-cubes-r186:${name}`,allowMath:true,generalControl:true,maxMemoryPages,maxIterations});
    const locals=parameters.filter(p=>p.kind==='local').map(p=>p.property);
    result.push(Object.freeze({name,...artifact,kernelSource,parameters:Object.freeze(parameters.map(p=>Object.freeze({...p}))),
      locals:Object.freeze(locals),color:parameters.some(p=>p.kind==='color'),
      sourceSpan:Object.freeze({start:loop.start,end:loop.end})}));
  }
  return Object.freeze(result);
}
