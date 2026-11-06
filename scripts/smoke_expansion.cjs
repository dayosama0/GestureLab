/* Expansion acceptance: real local checkpoints/API only, no fabricated results.
 * Requires all five production models and six saved experiment runs.
 * Does not save uploaded image bytes or image-bearing screenshots.
 */
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
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
const modelIds=['simple_cnn','resnet18','mobilenetv4','resnet18_finetuned','mobilenetv4_finetuned'];
const runIds=[...modelIds,'mobilenetv4_noaug'];
const labels=['fist','open_palm','peace','thumbs_up'];
const checks=[];
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
function passed(name,detail){checks.push({name,status:'PASS',detail});console.log('PASS '+name+(detail?' '+JSON.stringify(detail):''));}
function localIntegrity(){
 const legacy=JSON.parse(fs.readFileSync(path.join(evidence,'legacy_before_expansion.json'),'utf8'));
 for(const [relative,digest] of Object.entries(legacy))assert.equal(hash(path.join(root,...relative.split(/[\\/]/))),digest,'Legacy artifact changed: '+relative);
 passed('legacy checkpoints and metrics unchanged',{files:Object.keys(legacy).length});
 for(const id of modelIds.filter(id=>id.endsWith('_finetuned'))){
  const run=JSON.parse(fs.readFileSync(path.join(root,'artifacts','metrics',id+'.json'),'utf8'));
  assert.equal(run.checkpoint_sha256,hash(path.join(root,'artifacts','checkpoints',id+'.pt')));
  assert(run.metadata.first_backbone_conv_weight_l2_change>0,'No actual backbone optimization evidence');
  assert(run.metadata.validation_metrics&&Number.isFinite(run.metadata.validation_metrics.f1_macro));
  assert.equal(run.metadata.backbone_frozen,false);
  assert.equal(run.latency.device,'cpu','CPU serving benchmark must be labelled CPU');
  assert.equal(run.original_test.metrics.sample_count,60);
  assert.equal(run.metrics.accuracy,run.original_test.metrics.accuracy);
  for(const split of ['original_test','expanded_test']){
   const output=run[split];const matrix=output.confusion_matrix;
   assert.equal(matrix.length,4);assert(matrix.every(row=>row.length===4));
   assert.equal(matrix.flat().reduce((a,b)=>a+b,0),output.metrics.sample_count);
  }
  assert(run.expanded_test.metrics.sample_count>0);
  assert.equal(run.gradcam_verification.passed,true);
  assert.equal(run.gradcam_verification.real_gradients,true);
 }
 passed('new run hashes, true backbone changes, separated test counts and CPU timing');
}
(async()=>{
 fs.mkdirSync(evidence,{recursive:true});
 localIntegrity();
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'msedge'});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  const page=await context.newPage();const errors=[];page.on('pageerror',error=>errors.push(String(error)));
  const registryResponse=await page.request.get(base+'/api/models');assert.equal(registryResponse.status(),200);
  const registry=await registryResponse.json();assert.deepEqual(registry.classes,labels);
  assert.deepEqual(registry.models.map(m=>m.id).sort(),[...modelIds].sort());
  assert(registry.models.every(m=>m.available),'At least one genuine trained checkpoint unavailable');
  const runsResponse=await page.request.get(base+'/api/experiments');assert.equal(runsResponse.status(),200);
  const experiments=await runsResponse.json();assert.deepEqual(experiments.runs.map(r=>r.run_id).sort(),[...runIds].sort());
  for(const id of modelIds.filter(id=>id.endsWith('_finetuned'))){const run=experiments.runs.find(r=>r.model_id===id);assert.equal(run.original_test.metrics.sample_count,60);assert(run.expanded_test.metrics.sample_count>0);}
  passed('API exposes five real models and six factual experiments');
  const sampleDir=path.join(root,'data','ready_dataset','test','fist');const sample=path.join(sampleDir,fs.readdirSync(sampleDir).filter(n=>n.endsWith('.jpg')).sort()[0]);
  for(const model of registry.models){
   const response=await page.request.post(base+'/api/predict',{multipart:{file:{name:'heldout.jpg',mimeType:'image/jpeg',buffer:fs.readFileSync(sample)},model_id:model.id,explain:'true'}});
   assert.equal(response.status(),200,model.id+': '+await response.text());const result=await response.json();
   assert.equal(result.model_id,model.id);assert.equal(result.version,model.version);
   assert.deepEqual(Object.keys(result.probabilities).sort(),[...labels].sort());
   assert(Math.abs(Object.values(result.probabilities).reduce((a,b)=>a+b,0)-1)<1e-5);
   const size=model.metadata.input_size;assert.equal(result.preprocessing.input_size,size);
   assert.deepEqual(result.preprocessing.mean,model.metadata.mean);assert.deepEqual(result.preprocessing.std,model.metadata.std);
   assert.equal(result.explanation.width,size);assert.equal(result.explanation.height,size);
   const png=Buffer.from(result.explanation.image_base64,'base64');assert.equal(png.subarray(1,4).toString(),'PNG');assert.equal(png.readUInt32BE(16),size);assert.equal(png.readUInt32BE(20),size);
   if(model.id==='mobilenetv4_finetuned'){assert.equal(size,256);assert.deepEqual(model.metadata.mean,[.5,.5,.5]);assert.deepEqual(model.metadata.std,[.5,.5,.5]);}
   passed('actual checkpoint prediction and correctly sized Grad-CAM',{model_id:model.id,input_size:size,label:result.label,version:result.version});
  }
  await page.goto(base);await page.locator('.connection').filter({hasText:'5 models ready'}).waitFor();
  assert(await page.getByRole('button',{name:'Analyze hand region'}).isDisabled());
  assert.equal(await page.getByRole('combobox').inputValue(),'resnet18_finetuned','Current default must follow saved validation F1, not test scores');
  await page.getByLabel('Upload gesture image').setInputFiles(sample);await page.waitForFunction(()=>document.querySelector('img[alt="Uploaded image for hand crop"]')?.naturalWidth>0);
  await page.getByRole('combobox').selectOption('mobilenetv4_finetuned');
  const reply=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.getByRole('button',{name:'Analyze hand region'}).click();
  const response=await reply;assert.equal(response.status(),200,await response.text());const prediction=await response.json();assert.equal(prediction.preprocessing.input_size,256);
  await page.getByText('Inference complete',{exact:true}).waitFor();
  const heatmap=page.locator('.heat-overlay');await heatmap.waitFor();await heatmap.evaluate(async image=>{await image.decode();if(image.naturalWidth!==256||image.naturalHeight!==256)throw Error('Browser heatmap does not use native256');});
  await page.getByAltText('Exact prepared hand region used by the model').evaluate(async image=>{await image.decode();if(image.naturalWidth!==256||image.naturalHeight!==256)throw Error('Prepared base does not match native256 overlay');});
  await page.getByLabel('Overlay opacity').fill('40');assert.equal(await page.getByLabel('Overlay opacity').inputValue(),'40');
  passed('browser upload and native256 fine-tuned explanation');
  await page.getByRole('button',{name:'Compare',exact:true}).click();await page.getByRole('button',{name:'Compare available models'}).click();await page.waitForFunction(()=>document.querySelectorAll('.comparison-card .result-head').length===5,null,{timeout:60000});
  passed('five-model same-image browser comparison');
  await page.getByRole('button',{name:'Experiments',exact:true}).click();assert.equal(await page.locator('.metrics-table tbody tr').count(),6);
  await page.getByLabel('Inspect experiment').selectOption('mobilenetv4_finetuned');
  passed('six factual runs and selectable fine-tuned experiment');
  await page.setViewportSize({width:360,height:800});
  for(const name of ['Studio','Compare','Experiments','About']){await page.getByRole('button',{name,exact:true}).click();assert(await page.getByRole('heading',{level:1}).isVisible());const dim=await page.evaluate(()=>({view:innerWidth,document:document.documentElement.scrollWidth}));assert(dim.document<=dim.view+1,name+' page overflow '+JSON.stringify(dim));}
  assert.equal(errors.length,0,errors.join('\n'));passed('360px four-page layout and no uncaught browser errors');
  await context.close();
 }finally{await browser.close();}
})().catch(error=>{checks.push({name:'expansion acceptance',status:'FAIL',detail:String(error)});console.error(error);process.exitCode=1;}).finally(()=>fs.writeFileSync(path.join(evidence,'expansion_smoke.json'),JSON.stringify({time:new Date().toISOString(),checks},null,2)));
