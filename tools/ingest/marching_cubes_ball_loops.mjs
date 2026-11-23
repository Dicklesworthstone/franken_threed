/** Closed application addBall loops. Source scalar expressions stay JavaScript;
 * one guarded addon transaction consumes their private numeric records. */
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {numericLoopScopeBindings} from './numeric_loop_islands.mjs';

const MATH=new Set(['abs','acos','asin','atan','atan2','ceil','cos','exp','floor','log','max','min','pow','round','sign','sin','sqrt','tan','trunc']);
const fresh=(source,role)=>{let name='__f3d_balls_'+role;while(source.includes(name))name+='_';return name;};
const id=(node,name)=>node?.type==='Identifier'&&(name===undefined||node.name===name);
const property=node=>node?.type==='MemberExpression'&&!node.optional?
  (!node.computed&&id(node.property)?node.property.name:node.computed&&node.property.type==='Literal'?node.property.value:null):null;

export function specializeMarchingCubesBallLoops(source,{runtimeModule,maxLoops=32}={}) {
  const unchanged={changed:false,edits:[],header:'',report:null};
  if(!source.includes('addBall'))return unchanged;
  let ast;try{ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module',locations:true});}catch{return unchanged;}
  let evalCall=false;walk.simple(ast,{CallExpression(node){if(id(node.callee,'eval'))evalCall=true;}});
  if(evalCall)return unchanged;
  const bindings=numericLoopScopeBindings(),edits=[],reports=[];
  const attempt=fresh(source,'try'),register=fresh(source,'array'),recipe=fresh(source,'recipe');
  const headers=[];
  walk.fullAncestor(ast,(loop,_state,ancestors)=>{
    if(loop.type!=='ForStatement'||reports.length>=maxLoops||ancestors.at(-2)?.type==='LabeledStatement')return;
    const init=loop.init?.declarations?.[0],test=loop.test,update=loop.update;
    if(loop.init?.type!=='VariableDeclaration'||loop.init.kind!=='let'||loop.init.declarations.length!==1||
      !id(init.id)||init.init?.type!=='Literal'||init.init.value!==0||test?.type!=='BinaryExpression'||test.operator!=='<'||
      !id(test.left,init.id.name)||!id(test.right)||update?.type!=='UpdateExpression'||update.operator!=='++'||
      !id(update.argument,init.id.name)||loop.body.type!=='BlockStatement'||!loop.body.body.length)return;
    const declarations=loop.body.body.slice(0,-1),last=loop.body.body.at(-1);
    if(declarations.some(node=>node.type!=='VariableDeclaration'||node.kind!=='const'||
      node.declarations.some(d=>!id(d.id)||!d.init)))return;
    const locals=new Set([init.id.name,...declarations.flatMap(node=>node.declarations.map(d=>d.id.name))]);
    if(locals.has('Math')||locals.has(test.right.name))return;
    const captures=new Set([test.right.name]),math=new Set(),changes=[];
    let owner=null,palette=null,condition=null,valid=true;
    const numeric=node=>{
      if(!node)return false;
      if(node.type==='Literal')return typeof node.value==='number';
      if(id(node)){if(!locals.has(node.name))captures.add(node.name);return node.name!=='arguments';}
      if(node.type==='UnaryExpression'&&['+','-','~'].includes(node.operator))return numeric(node.argument);
      if(node.type==='BinaryExpression'&&['+','-','*','/','%','**','&','|','^','<<','>>','>>>'].includes(node.operator))
        return numeric(node.left)&&numeric(node.right);
      if(node.type==='CallExpression'&&!node.optional&&id(node.callee?.object,'Math')&&MATH.has(property(node.callee))) {
        math.add(property(node.callee));return node.arguments.every(numeric);
      }
      return false;
    };
    for(const declaration of declarations)for(const d of declaration.declarations)valid&&=numeric(d.init);
    const records=fresh(source,'records'),inputs=fresh(source,'inputs'),colors=fresh(source,'colors'),colorCount=fresh(source,'color_count');
    const slot=fresh(source,'slot'),args=fresh(source,'args'),index=fresh(source,'color_index');
    const call=statement=>{
      if(statement?.type==='BlockStatement'&&statement.body.length===1)statement=statement.body[0];
      const node=statement?.type==='ExpressionStatement'?statement.expression:null;
      if(node?.type!=='CallExpression'||node.optional||property(node.callee)!=='addBall'||!id(node.callee.object)||
        ![5,6].includes(node.arguments.length)||node.arguments.slice(0,5).some(arg=>!numeric(arg)))return false;
      if(owner&&owner!==node.callee.object.name)return false;
      owner=node.callee.object.name;
      let color='';
      if(node.arguments.length===6) {
        const value=node.arguments[5];
        if(value.type!=='MemberExpression'||value.optional||!value.computed||!id(value.object)||!numeric(value.property)||
          (palette&&palette!==value.object.name))return false;
        palette=value.object.name;
        color='const '+index+'='+source.slice(value.property.start,value.property.end)+';'+
          'if('+index+'<0||'+index+'>='+colorCount+'||'+index+'%1!==0)return false;';
      }
      const values=[0,1,2,3,4].map(i=>args+'['+i+']').concat(node.arguments.length===6?
        [0,1,2].map(i=>colors+'['+index+'*3+'+i+']'):[0,1,2].map(i=>args+'['+i+']'));
      changes.push({start:statement.start,end:statement.end,text:'{if(!(()=>{const '+args+'=['+
        node.arguments.slice(0,5).map(arg=>source.slice(arg.start,arg.end)).join(',')+'];'+color+
        values.map(value=>records+'['+slot+'++]='+value+';').join('')+'return true;})())return false;}'});
      return true;
    };
    if(last.type==='IfStatement') {
      const test=last.test;
      if(test.type!=='BinaryExpression'||!['===','!=='].includes(test.operator)||!id(test.left)||
        test.right.type!=='Literal'||typeof test.right.value!=='string'||locals.has(test.left.name))return;
      condition=source.slice(test.start,test.end);
      if(!call(last.consequent)||!call(last.alternate))return;
      changes.push({start:test.start,end:test.end,text:inputs+'['+captures.size+']'});
    }else if(!call(last))return;
    if(!valid||locals.has(owner)||locals.has(palette)||captures.has(owner)||captures.has(palette))return;
    const scopes=ancestors.slice(0,-1),bound=name=>name!=='arguments'&&
      [...scopes].reverse().some(scope=>bindings(scope,loop).has(name));
    if(![owner,...captures,...(palette?[palette]:[])].every(bound)||
      (condition&&!bound(last.test.left.name))||math.size&&bound('Math'))return;
    let body=source.slice(loop.start,loop.end);
    for(const change of changes.sort((a,b)=>b.start-a.start))
      body=body.slice(0,change.start-loop.start)+change.text+body.slice(change.end-loop.start);
    const n=reports.length,prepare=fresh(source,'prepare_'+n),token=fresh(source,'token_'+n),captured=fresh(source,'capture_'+n);
    const countIndex=[...captures].indexOf(test.right.name);
    // Function declarations can be called through an ESM cycle before this
    // module evaluates. An unregistered var token retains the original loop.
    headers.push('var '+prepare+'=('+records+','+inputs+','+colors+','+colorCount+')=>{'+
      [...captures].map((name,i)=>'const '+name+'='+inputs+'['+i+'];').join('')+'let '+slot+'=0;'+body+'return true;};\n'+
      'var '+token+'='+recipe+'('+prepare+','+captures.size+','+countIndex+','+JSON.stringify([...math])+','+!!condition+');\n');
    const values=[...captures,...(condition?['('+condition+')']:[])];
    edits.push({start:loop.start,end:loop.end,text:'{let '+captured+';try{'+captured+'=['+owner+', ['+
      values.join(',')+'], '+(palette??'null')+'];}catch{}\nif(!'+captured+'||!'+attempt+'('+token+','+
      captured+'[0],'+captured+'[1],'+captured+'[2])){'+source.slice(loop.start,loop.end)+'}}'});
    reports.push({sourceSpan:{start:loop.start,end:loop.end,line:loop.loc.start.line,column:loop.loc.start.column},
      palette:!!palette,retainedMath:[...math]});
  });
  if(!reports.length)return unchanged;
  walk.simple(ast,{ArrayExpression(node){if(edits.some(edit=>edit.start<=node.start&&edit.end>=node.end))return;
    edits.push({start:node.start,end:node.start,text:'('+register+'('},{start:node.end,end:node.end,text:'))'});}});
  return {changed:true,edits,header:'import {tryMarchingCubesBallLoop as '+attempt+',registerMarchingCubesBallArray as '+register+
    ',createMarchingCubesBallRecipe as '+recipe+'} from '+JSON.stringify(runtimeModule)+';\n'+headers.join(''),
    report:{kind:'source-marching-cubes-ball-loops',compiledLoops:reports.length,sourceDerived:true,
      accelerationClaim:false,loops:reports}};
}
