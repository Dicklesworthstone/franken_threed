/** Test-only host services; JavaScript parsing/execution and threads are native.
 * Classic scripts run with V8's global Script goal; ESM uses Node's native loader.
 * This models file-backed importScripts/fetch, not browser HTTP/CORS/SharedWorker.
 */
import fs from 'node:fs';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {parentPort, workerData, Worker as Thread} from 'node:worker_threads';

const {entry, workerType} = workerData;
globalThis.self = globalThis;
globalThis.location = new URL(entry);
globalThis.name = workerData.name ?? '';
globalThis.postMessage = (value, transfer) => parentPort.postMessage(value, transfer);
let onmessage;
Object.defineProperty(globalThis, 'onmessage', {configurable:true, set(value){onmessage=value;}});
const messages=[];let ready=false;
parentPort.on('message', data => {if(ready)onmessage?.({data});else messages.push(data);});
function script(url) {
  const source=fs.readFileSync(new URL(url),'utf8');
  new vm.Script(source,{filename:url,importModuleDynamically:(specifier,_script,attributes)=>
    import(new URL(specifier,url).href,{with:attributes})}).runInThisContext();
}
globalThis.importScripts = (...values) => {
  if(workerType==='module')throw new TypeError('importScripts is not available in module workers');
  const urls=values.map(value=>new URL(String(value),entry).href);
  for(const url of urls)script(url);
};
globalThis.fetch = async input => {
  const url=new URL(String(input),entry);
  if(url.protocol!=='file:')throw new Error('Test host never performs network fetches');
  return new Response(await readFile(url),{headers:{'content-type':url.pathname.endsWith('.wasm')?'application/wasm':'application/octet-stream'}});
};
globalThis.Worker = class {
  constructor(url,options={}) {
    this.thread=new Thread(new URL(import.meta.url),{workerData:{entry:new URL(String(url),entry).href,
      workerType:options?.type??'classic',name:options?.name??''},execArgv:['--experimental-vm-modules']});
    this.thread.on('error',error=>{if(this.onerror)this.onerror(error);else throw error;});
    this.thread.on('message',data=>this.onmessage?.({data}));
  }
  postMessage(value,transfer){this.thread.postMessage(value,transfer);}
  terminate(){return this.thread.terminate();}
};
if(workerType==='classic')script(entry);else await import(entry);
ready=true;for(const data of messages)onmessage?.({data});
