/* Genuine private local browser recording. No API fixtures or generated frames.
 * Requires running localhost app, actual checkpoints and user-provided dataset.
 */
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const {spawnSync}=require('node:child_process');
function loadPlaywright(){
 if(process.env.PLAYWRIGHT_MODULE)return require(process.env.PLAYWRIGHT_MODULE);
 try{return require('../frontend/node_modules/playwright');}
 catch(error){if(error.code!=='MODULE_NOT_FOUND')throw error;}
 return require('C:/Users/Ali/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
}
const {chromium}=loadPlaywright();
const root=path.resolve(__dirname,'..');
const output=path.join(root,'artifacts','demo');
const base=process.env.GESTURELAB_URL||'http://127.0.0.1:5173';
fs.mkdirSync(output,{recursive:true});
const notes=[];let start=0;
const timestamp=seconds=>{const n=Math.floor(seconds*1000);return `${String(Math.floor(n/3600000)).padStart(2,'0')}:${String(Math.floor(n/60000)%60).padStart(2,'0')}:${String(Math.floor(n/1000)%60).padStart(2,'0')},${String(n%1000).padStart(3,'0')}`;};
function saveNotes(){fs.writeFileSync(path.join(output,'gesturelab-demo.srt'),notes.map((n,i)=>`${i+1}\n${timestamp(n.start)} --> ${timestamp(n.end)}\n${n.text}\n`).join('\n'));}
async function stage(end,text,action){const began=(Date.now()-start)/1000;console.log(`RECORD ${began.toFixed(1)}s ${text}`);notes.push({start:began,end,text});await action();saveNotes();const remaining=end*1000-(Date.now()-start);if(remaining>0)await new Promise(resolve=>setTimeout(resolve,remaining));}
(async()=>{
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'msedge'});
 let context,video,videoPath;
 try{
  context=await browser.newContext({viewport:{width:1280,height:800},recordVideo:{dir:output,size:{width:1280,height:800}}});
  start=Date.now();const page=await context.newPage();video=page.video();
  await page.goto(base);await page.locator('.connection').filter({hasText:'3 models ready'}).waitFor();
  const registry=await (await page.request.get(base+'/api/models')).json();assert.equal(registry.models.filter(m=>m.available).length,3);
  const experiments=await (await page.request.get(base+'/api/experiments')).json();assert.equal(experiments.runs.length,4);
  const sampleDir=path.join(root,'data','ready_dataset','test','fist');const sample=path.join(sampleDir,fs.readdirSync(sampleDir).filter(n=>n.endsWith('.jpg')).sort()[0]);
  await stage(18,'GestureLab AI: a real local four-class hand-gesture research app. Three trained models are ready. This is a silent browser recording with sidecar captions.',async()=>{
   await page.getByRole('button',{name:'Studio',exact:true}).click();await page.mouse.move(980,280);
  });
  await stage(36,'Upload a private held-out fist photograph. The manual crop keeps the full hand in view; preprocessing uses RGB and 224 by 224 input.',async()=>{
   await page.getByLabel('Upload gesture image').setInputFiles(sample);await page.waitForFunction(()=>document.querySelector('img[alt="Uploaded image for hand crop"]')?.naturalWidth>0);
   await page.getByLabel('Crop size',{exact:true}).fill('90');await page.getByLabel('Horizontal crop position').fill('55');
   await page.getByLabel('Crop size',{exact:true}).fill('100');await page.getByLabel('Horizontal crop position').fill('50');
   await page.getByRole('combobox').selectOption('mobilenetv4');await page.getByRole('button',{name:'Analyze hand region'}).scrollIntoViewIfNeeded();
  });
  await stage(56,'Run the actual MobileNetV4 checkpoint. The API returns a label, all four softmax probabilities, measured inference time and the checkpoint version.',async()=>{
   const reply=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.getByRole('button',{name:'Analyze hand region'}).click();
   const response=await reply;assert.equal(response.status(),200,await response.text());const result=await response.json();
   fs.writeFileSync(path.join(output,'recorded-prediction.json'),JSON.stringify({model_id:result.model_id,label:result.label,version:result.version,probabilities:result.probabilities,latency_ms:result.latency_ms},null,2));
   await page.getByText('Inference complete',{exact:true}).waitFor();await page.locator('.result').scrollIntoViewIfNeeded();
  });
  await stage(78,'Grad-CAM is calculated from the selected model activations and gradients. Adjust overlay opacity and compare with the original ROI. Attribution is approximate, not a correctness guarantee.',async()=>{
   await page.locator('.explain-panel').scrollIntoViewIfNeeded();await page.getByAltText('Gradient based class activation explanation').waitFor();
   await page.getByLabel('Overlay opacity').fill('20');await new Promise(r=>setTimeout(r,3000));await page.getByLabel('Overlay opacity').fill('85');
   await new Promise(r=>setTimeout(r,3000));await page.getByLabel('Show explanation',{exact:true}).uncheck();await new Promise(r=>setTimeout(r,2000));await page.getByLabel('Show explanation',{exact:true}).check();
  });
  await stage(98,'Camera is requested through the real browser. If this automated browser has no allowed camera, the app presents its actual error and upload remains the fallback.',async()=>{
   await page.getByRole('button',{name:'Use camera'}).scrollIntoViewIfNeeded();await page.getByRole('button',{name:'Use camera'}).click();
   const outcome=await Promise.race([page.getByRole('alert').waitFor().then(async()=>({state:'error',text:await page.getByRole('alert').innerText()})),page.getByRole('button',{name:'Capture hand region'}).waitFor().then(()=>({state:'preview',text:'Real camera preview available; no physical hand performance claim'}))]);
   fs.writeFileSync(path.join(output,'camera-outcome.json'),JSON.stringify(outcome,null,2));console.log('CAMERA',JSON.stringify(outcome));
   if(outcome.state==='error')await page.getByRole('alert').scrollIntoViewIfNeeded();
   else await page.getByRole('button',{name:'Stop camera'}).click();
  });
  await stage(122,'Compare the exact same prepared photograph with A1 SimpleCNN, frozen ImageNet ResNet18 and modern MobileNetV4. Predictions and latency are actual API results.',async()=>{
   if(await page.getByLabel('Dismiss error').count())await page.getByLabel('Dismiss error').click();
   await page.getByRole('button',{name:'Compare',exact:true}).click();await page.getByRole('button',{name:'Compare available models'}).click();
   await page.waitForFunction(()=>document.querySelectorAll('.comparison-card .result-head').length===3);await page.locator('.comparison-grid').scrollIntoViewIfNeeded();
  });
  await stage(146,'Experiments load saved measurements from the fixed 60-image test split: baseline comparison plus the MobileNetV4 augmentation ablation. Validation selected checkpoints; clean test images were not augmented.',async()=>{
   await page.getByRole('button',{name:'Experiments',exact:true}).click();await page.locator('.metrics-table').scrollIntoViewIfNeeded();assert.equal(await page.locator('.metrics-table tbody tr').count(),4);
  });
  await stage(168,'Inspect MobileNetV4 without augmentation: real training and validation curves, class confusion and separate dim-light, Gaussian-blur and JPEG stress scores.',async()=>{
   await page.getByLabel('Inspect experiment').selectOption('mobilenetv4_noaug');await page.locator('.chart-panel').first().scrollIntoViewIfNeeded();await page.getByText('Environment stress tests',{exact:true}).scrollIntoViewIfNeeded();
  });
  await stage(190,'Switch to moderate training-only augmentation. This ablation uses matched optimizer settings; augmentation expands training exposure. These perturbations are not ArtAug or synthetic video.',async()=>{
   await page.getByLabel('Inspect experiment').selectOption('mobilenetv4');await page.locator('.chart-panel').first().scrollIntoViewIfNeeded();await page.getByText('Environment stress tests',{exact:true}).scrollIntoViewIfNeeded();
  });
  await stage(214,'Methodology and limits: modern MobileNetV4 provenance, local processing, uncalibrated confidence, tiny test set and unverified person separation. Dataset consent and redistribution require author confirmation.',async()=>{
   await page.getByRole('button',{name:'About',exact:true}).click();await page.getByRole('heading',{name:'From A1 to modern CV'}).scrollIntoViewIfNeeded();await new Promise(r=>setTimeout(r,8000));await page.getByRole('heading',{name:'Privacy, consent & limitations'}).scrollIntoViewIfNeeded();
  });
  await context.close();context=null;videoPath=await video.path();const finalPath=path.join(output,'gesturelab-demo.webm');
  if(videoPath!==finalPath)fs.renameSync(videoPath,finalPath);
  const ffmpeg=process.env.FFMPEG_PATH||'C:/Users/Ali/AppData/Local/ms-playwright/ffmpeg-1013/ffmpeg-win64.exe';
  const probe=spawnSync(ffmpeg,['-hide_banner','-i',finalPath],{encoding:'utf8'});fs.writeFileSync(path.join(output,'video-probe.txt'),probe.stderr||probe.stdout||'');
  const duration=(probe.stderr||'').match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
  const seconds=duration?Number(duration[1])*3600+Number(duration[2])*60+Number(duration[3]):null;
  assert(seconds!==null&&seconds>=180&&seconds<=300,`Invalid recorded duration: ${seconds}`);
  notes[notes.length-1].end=seconds;
  const manifest={file:finalPath,duration_seconds:seconds,width:1280,height:800,kind:'Actual local Playwright browser recording',audio:'Silent; no narration',captions:'gesturelab-demo.srt',privacy:'Contains private supplied dataset photograph; local only, do not publish without consent',camera:JSON.parse(fs.readFileSync(path.join(output,'camera-outcome.json'),'utf8')),captured_at:new Date().toISOString(),stages:notes};
  fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));console.log('VERIFIED',JSON.stringify({file:finalPath,duration_seconds:seconds,bytes:fs.statSync(finalPath).size}));
 }finally{if(context)await context.close();await browser.close();saveNotes();}
})().catch(error=>{console.error(error);process.exitCode=1;});
