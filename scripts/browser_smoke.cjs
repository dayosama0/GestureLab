/* Local browser acceptance checks. Missing-weight/camera fixtures are browser-only.
 * Uses real API/checkpoints for upload, explanation, comparison and experiments.
 * No private source image screenshots are saved.
 */
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
function loadPlaywright(){
 if(process.env.PLAYWRIGHT_MODULE)return require(process.env.PLAYWRIGHT_MODULE);
 try{return require('../frontend/node_modules/playwright');}
 catch(error){if(error.code!=='MODULE_NOT_FOUND')throw error;}
 return require('C:/Users/Ali/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
}
const {chromium}=loadPlaywright();
const root=path.resolve(__dirname,'..');
const base=process.env.GESTURELAB_URL||'http://127.0.0.1:5173';
const evidence=path.join(root,'artifacts','verification');
fs.mkdirSync(evidence,{recursive:true});
const checks=[];
function passed(name,detail){checks.push({name,status:'PASS',detail});console.log(`PASS ${name}${detail?' '+JSON.stringify(detail):''}`);}
function skipped(name,detail){checks.push({name,status:'SKIP',detail});console.log(`SKIP ${name} ${detail}`);}
(async()=>{
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'msedge'});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  const page=await context.newPage();
  const errors=[];page.on('pageerror',error=>errors.push(String(error)));
  await page.goto(base);await page.getByRole('button',{name:'Analyze hand region'}).waitFor();
  await page.waitForFunction(()=>!document.body.textContent.includes('Connecting…'));
  assert(await page.getByRole('button',{name:'Analyze hand region'}).isDisabled());
  passed('empty Studio requires source image');
  await page.screenshot({path:path.join(evidence,'studio-empty-desktop.png'),fullPage:true});
  const modelResponse=await page.request.get(base+'/api/models');assert.equal(modelResponse.status(),200);
  const registry=await modelResponse.json();const available=registry.models.filter(m=>m.available);
  const testDir=path.join(root,'data','ready_dataset','test','fist');
  const sample=fs.existsSync(testDir)?fs.readdirSync(testDir).find(n=>n.endsWith('.jpg')):null;
  if(available.length&&sample){
   await page.getByLabel('Upload gesture image').setInputFiles(path.join(testDir,sample));
   await page.getByAltText('Uploaded image for hand crop').waitFor();
   await page.waitForFunction(()=>document.querySelector('img[alt="Uploaded image for hand crop"]')?.naturalWidth>0);
   const selected=available.find(m=>m.id==='mobilenetv4')||available[0];
   await page.getByRole('combobox').selectOption(selected.id);
   const reply=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');
   await page.getByRole('button',{name:'Analyze hand region'}).click();
   const response=await reply;assert.equal(response.status(),200,await response.text());const prediction=await response.json();
   assert.equal(prediction.model_id,selected.id);assert(Math.abs(Object.values(prediction.probabilities).reduce((a,b)=>a+b,0)-1)<1e-5);
   await page.getByText('Inference complete',{exact:true}).waitFor();
   await page.getByAltText('Gradient based class activation explanation').waitFor();
   await page.getByLabel('Overlay opacity').fill('30');
   assert.equal(await page.getByLabel('Overlay opacity').inputValue(),'30');
   passed('real upload prediction and Grad-CAM',{model_id:selected.id,label:prediction.label,version:prediction.version});
   await page.getByRole('button',{name:'Compare',exact:true}).click();
   await page.getByRole('button',{name:'Compare available models'}).click();
   await page.waitForFunction(n=>document.querySelectorAll('.comparison-card .result-head').length===n,available.length,{timeout:60000});
   passed('same-image model comparison',{models:available.map(m=>m.id)});
  }else skipped('real checkpoint upload/Grad-CAM/comparison','No available trained checkpoint or source test image');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  const expResponse=await page.request.get(base+'/api/experiments');assert.equal(expResponse.status(),200);const experiments=await expResponse.json();
  if(experiments.runs.length){await page.getByText('Clean test comparison',{exact:true}).waitFor();assert.equal(await page.locator('.metrics-table tbody tr').count(),experiments.runs.length);passed('factual experiment rows',{runs:experiments.runs.map(r=>r.run_id)});}else{await page.getByText('Not evaluated yet',{exact:true}).waitFor();passed('unevaluated experiments empty state');}
  await page.setViewportSize({width:360,height:800});
  for(const name of ['Studio','Compare','Experiments','About']){
   await page.getByRole('button',{name,exact:true}).click();
   assert(await page.getByRole('heading',{level:1}).isVisible());
   const dimensions=await page.evaluate(()=>({view:innerWidth,document:document.documentElement.scrollWidth}));
   assert(dimensions.document<=dimensions.view+1,`${name} page overflows viewport ${JSON.stringify(dimensions)}`);
  }
  passed('360px responsive navigation and no page overflow');
  assert.equal(errors.length,0,errors.join('\n'));passed('no uncaught browser runtime errors');
  await context.close();
  const absent=await browser.newContext({viewport:{width:360,height:800}});const missingPage=await absent.newPage();
  await missingPage.route('**/api/models',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({classes:['fist','open_palm','peace','thumbs_up'],models:[{id:'simple_cnn',name:'A1 SimpleCNN',available:false,reason:'QA missing-checkpoint browser fixture'}]})}));
  await missingPage.route('**/api/experiments',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({available:false,runs:[]})}));
  await missingPage.addInitScript(()=>Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:()=>Promise.reject(new DOMException('QA denied','NotAllowedError'))}}));
  await missingPage.goto(base);await missingPage.locator('.connection').filter({hasText:'Models unavailable'}).waitFor();
  assert(await missingPage.getByRole('combobox').isDisabled());assert(await missingPage.getByRole('button',{name:'Analyze hand region'}).isDisabled());
  assert.equal(await missingPage.getByText('Inference complete',{exact:true}).count(),0);
  passed('missing checkpoints disable prediction (browser-only fixture)');
  await missingPage.getByRole('button',{name:'Use camera'}).click();
  await missingPage.getByRole('alert').filter({hasText:'Camera permission denied'}).waitFor();passed('camera denial handled (browser-only rejection)');
  await missingPage.screenshot({path:path.join(evidence,'camera-denied-mobile.png'),fullPage:true});
  await absent.close();
 }finally{await browser.close();fs.writeFileSync(path.join(evidence,'browser_smoke.json'),JSON.stringify({time:new Date().toISOString(),checks},null,2));}
})().catch(error=>{console.error(error);process.exitCode=1;});
