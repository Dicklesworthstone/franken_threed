/** Real browser proof for CLI-produced glTF poses, not a rendering benchmark.
 * F3D_CHROMIUM=/path/to/chromium node tools/ingest/animation.browser.mjs
 * The test remaps the package's single relative import to a data URL so managed
 * hosts can use CDP document injection. Production source and definitions run
 * unchanged; this does not certify direct file navigation or a Three.js mixer.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawn,spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {hierarchyAnimationFixture,glbFixture} from './fixtures/animation/gltf_fixture.mjs';
const browser=process.env.F3D_CHROMIUM;if(!browser)throw new Error('Set F3D_CHROMIUM to an installed Chromium executable');
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-pose-browser-')),entry=path.join(dir,'actor.glb'),out=path.join(dir,'player');
const fixture=hierarchyAnimationFixture(64);fs.writeFileSync(entry,glbFixture(fixture.model,fixture.bytes));
const built=spawnSync(process.execPath,[fileURLToPath(new URL('./cli.mjs',import.meta.url)),'--entry',entry,'--build-animation',out],{encoding:'utf8'});
assert.equal(built.status,0,built.stderr);fs.renameSync(entry,entry+'.not-used');
const encoded=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const runtime=encoded(fs.readFileSync(path.join(out,'animation_runtime.mjs')));
const module=encoded(fs.readFileSync(path.join(out,'animation.mjs'),'utf8').replace("'./animation_runtime.mjs'",JSON.stringify(runtime)));
const script=`import {createPlayer} from ${JSON.stringify(module)};
  const fail=message=>{throw new Error(message);};
  const player=createPlayer(),other=createPlayer(),palette=player.jointMatrices,world=player.worldMatrices;
  let checks=0,vertexChecks=0;
  for(let frame=0;frame<60;frame++) {
    const time=frame/30,u=time/2,x=(1-u)*10+u*12,y=u*2;
    player.sample(time);let wx=0,wy=0;
    for(let joint=0;joint<64;joint++) {
      wx+=x;wy+=y;
      const ex=Math.fround(wx-10*(joint+1)-5),ey=Math.fround(wy),o=joint*16;
      if(palette[o+12]!==ex||palette[o+13]!==ey)fail('Palette differs from independent translation-chain reference');checks++;
    }
    // Four-influence positions consume the generated palette. This is a CPU
    // functional reference, not a GPU draw or a claim of Wasm skinning here.
    for(let vertex=0;vertex<10000;vertex++) {
      let px=0,py=0,ex=0,ey=0;const vx=(vertex%11)/10,vy=(vertex%7)/5;
      for(let influence=0;influence<4;influence++) {
        const joint=(vertex+influence)%64,weight=[0.125,0.25,0.375,0.25][influence],o=joint*16;
        let ax=0,ay=0;for(let ancestor=0;ancestor<=joint;ancestor++){ax+=x;ay+=y;}
        px+=weight*(palette[o]*vx+palette[o+4]*vy+palette[o+12]);
        py+=weight*(palette[o+1]*vx+palette[o+5]*vy+palette[o+13]);
        ex+=weight*(vx+Math.fround(ax-10*(joint+1)-5));ey+=weight*(vy+Math.fround(ay));
      }
      if(Math.fround(px)!==Math.fround(ex)||Math.fround(py)!==Math.fround(ey))fail('Weighted position differs');vertexChecks++;
    }
    if(Math.abs(player.morphWeights[0]-u)>1e-12||Math.abs(player.morphWeights[1]-(1-u))>1e-12)fail('Morph weights differ');
    other.sample(2-time);if(player.jointMatrices!==palette||player.worldMatrices!==world)fail('Output identity changed');
  }
  player.sample(-1,{loop:true});if(player.time!==1||palette[12]!==-4)fail('Negative looping seek failed');
  player.sample(1,{clip:1});if(Math.abs(player.rotations[10]-Math.SQRT1_2)>1e-12)fail('Quaternion track failed');
  player.reset();if(player.morphWeights[0]!==0.25||palette[12]!==-5)fail('Reset failed');
  const version=player.version;let threw=false;try{player.sample(NaN);}catch{threw=true;}
  if(!threw||version!==player.version)fail('Invalid sample was published');
  player.dispose();threw=false;try{player.sample(0);}catch{threw=true;}if(!threw)fail('Disposed player ran');
  globalThis.result={frames:60,joints:64,paletteChecks:checks,weightedVertexChecks:vertexChecks,
    morphTracks:true,quaternionTracks:true,independentPlayers:true,stableOutputs:true,reverseAndLoop:true,transactionalFailure:true,disposed:true};`;
const html='<!doctype html><meta charset="utf-8"><title>glTF pose test</title><script>globalThis.WebAssembly=undefined;</script><script type="module" src="'+encoded(script)+'"></script>';
if(process.env.F3D_HTML_SAMPLE)fs.writeFileSync(process.env.F3D_HTML_SAMPLE,html);
const child=spawn(browser,['--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-proxy-server','--host-resolver-rules=MAP * ~NOTFOUND','--disable-background-networking','--user-data-dir='+path.join(dir,'profile'),'--remote-debugging-port=0','about:blank']);
let socket,timer;
try{
  const address=await new Promise((resolve,reject)=>{let log='';timer=setTimeout(()=>reject(new Error(log)),10000);child.on('error',reject);child.on('exit',code=>reject(new Error('Browser exited: '+code+' '+log)));child.stderr.on('data',bytes=>{log+=bytes;const match=/DevTools listening on (ws:\/\/[^\s]+)/.exec(log);if(match){clearTimeout(timer);resolve(match[1]);}});});
  socket=new WebSocket(address);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  let sequence=0;const pending=new Map(),errors=[],network=[];
  socket.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Network.requestWillBeSent')network.push(m.params.request.url);});
  function command(method,params={},sessionId){const id=++sequence;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method));},15000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,sessionId}));});}
  const {targetId}=await command('Target.createTarget',{url:'about:blank'}),{sessionId}=await command('Target.attachToTarget',{targetId,flatten:true});
  for(const domain of ['Page','Runtime','Network'])await command(domain+'.enable',{},sessionId);
  await command('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0},sessionId);
  const {frameTree}=await command('Page.getFrameTree',{},sessionId);await command('Page.setDocumentContent',{frameId:frameTree.frame.id,html},sessionId);
  let result;for(let tries=0;tries<200;tries++){result=(await command('Runtime.evaluate',{expression:'globalThis.result',returnByValue:true},sessionId)).result.value;if(result||errors.length)break;await new Promise(resolve=>setTimeout(resolve,50));}
  assert.deepEqual(errors,[],JSON.stringify(errors));assert.equal(result?.paletteChecks,3840);assert.equal(result?.weightedVertexChecks,600000);
  const external=network.filter(url=>!url.startsWith('data:')&&url!=='about:blank');assert.deepEqual(external,[]);
  console.log(JSON.stringify({browser:spawnSync(browser,['--version'],{encoding:'utf8'}).stdout.trim(),transport:'offline CDP document injection; source model relocated; Wasm unavailable',result,externalRequests:external},null,2));
  await command('Browser.close');
}finally{clearTimeout(timer);socket?.close();child.kill();}
