import {expect,test} from '@playwright/test';
// These fixtures exercise UI states only; production API routes contain no fixtures.
test.beforeEach(async({page})=>{
 await page.route('**/api/models',route=>route.fulfill({json:{classes:['fist','open_palm','peace','thumbs_up'],models:[{id:'simple_cnn',name:'SimpleCNN',available:false,reason:'Test: no checkpoint'}]}}));
 await page.route('**/api/experiments',route=>route.fulfill({json:{available:false,runs:[]}}));
});
test('missing weights and factual experiments remain usable without horizontal overflow',async({page})=>{
 await page.goto('/');await expect(page.getByRole('heading',{name:'Gesture recognition'})).toBeVisible();
 await expect(page.getByRole('button',{name:'Analyze hand region'})).toBeDisabled();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
 await page.getByRole('button',{name:'Experiments',exact:true}).click();await expect(page.getByText('Not evaluated yet')).toBeVisible();
 await page.getByRole('button',{name:'Compare',exact:true}).click();await expect(page.getByText('Test: no checkpoint')).toBeVisible();
 await page.getByRole('button',{name:'About',exact:true}).click();await expect(page.getByRole('heading',{name:'Privacy, consent & limitations'})).toBeVisible();
});
test('camera permission failure has an upload fallback',async({page})=>{
 await page.addInitScript(()=>{Object.defineProperty(navigator,'mediaDevices',{value:{getUserMedia:()=>Promise.reject(new DOMException('Permission denied','NotAllowedError'))}});});
 await page.goto('/');await page.getByRole('button',{name:'Use camera'}).click();await expect(page.getByRole('alert')).toContainText('Camera permission denied');await expect(page.getByRole('button',{name:'Choose image'})).toBeVisible();
});
test('late camera permission cannot leave a stream running after navigation',async({page})=>{
 await page.addInitScript(()=>{const state={stopped:false};Object.assign(window,{cameraTest:state});Object.defineProperty(navigator,'mediaDevices',{value:{getUserMedia:()=>new Promise(resolve=>setTimeout(()=>resolve({getTracks:()=>[{stop:()=>{state.stopped=true;}}]}),300))}});});
 await page.goto('/');await page.getByRole('button',{name:'Use camera'}).click();await page.getByRole('button',{name:'About',exact:true}).click();await expect.poll(()=>page.evaluate(()=>(window as unknown as {cameraTest:{stopped:boolean}}).cameraTest.stopped)).toBe(true);
});
test('native crop resolution and transparent gray compositing survive browser encoding',async({page})=>{
 await page.goto('/');
 const prepared=await page.evaluate(async()=>{
  const {cropImage}=await import(window.location.origin+'/src/image.ts');
  const source=document.createElement('canvas');source.width=640;source.height=480;
  const blob=await cropImage(source,640,480,{x:50,y:50,size:75});
  const bitmap=await createImageBitmap(blob);const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
  const context=canvas.getContext('2d')!;context.drawImage(bitmap,0,0);const pixel=Array.from(context.getImageData(0,0,1,1).data);bitmap.close();
  return {width:canvas.width,height:canvas.height,pixel,type:blob.type};
 });
 expect(prepared).toEqual({width:360,height:360,pixel:[128,128,128,255],type:'image/png'});
});

test('portrait hand crops preserve aspect and expose shape controls',async({page})=>{
 await page.goto('/');
 const sample=Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="white"/></svg>');
 // Generate a real PNG in the browser; fixture images never reach production model metrics.
 const png=await page.evaluate(async(svg)=>{
  const source=new Image();source.src='data:image/svg+xml;base64,'+btoa(svg);await source.decode();
  const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;canvas.getContext('2d')!.drawImage(source,0,0);
  return canvas.toDataURL('image/png').split(',')[1];
 },sample.toString());
 await page.getByLabel('Upload gesture image').setInputFiles({name:'test.png',mimeType:'image/png',buffer:Buffer.from(png,'base64')});
 await page.getByRole('button',{name:'Tall hand crop',exact:true}).click();
 await expect(page.getByRole('button',{name:'Tall hand crop',exact:true})).toHaveAttribute('aria-pressed','true');
 const dimensions=await page.evaluate(async()=>{
  const {cropImage}=await import(window.location.origin+'/src/image.ts');const source=document.createElement('canvas');source.width=640;source.height=480;
  const bitmap=await createImageBitmap(await cropImage(source,640,480,{x:50,y:50,size:75,aspect:.65}));
  const result={width:bitmap.width,height:bitmap.height};bitmap.close();return result;
 });
 expect(dimensions).toEqual({width:234,height:360});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});
