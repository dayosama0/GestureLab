export type Crop = {x:number;y:number;size:number;aspect?:number};
// Keep detail for each model's own resize while bounding an in-memory PNG upload.
export const MAX_ROI_EDGE = 1536;
export function cropRectangle(width:number,height:number,crop:Crop){
 const aspect=crop.aspect??1;
 if(![width,height,crop.x,crop.y,crop.size,aspect].every(Number.isFinite)||width<=0||height<=0||aspect<=0||crop.size<=0||crop.size>100||crop.x<0||crop.x>100||crop.y<0||crop.y>100)throw new Error('Image dimensions must be positive and crop must stay within the image.');
 const baseWidth=Math.min(width,height*aspect);
 const cropWidth=baseWidth*crop.size/100;const cropHeight=cropWidth/aspect;
 return {left:(width-cropWidth)*crop.x/100,top:(height-cropHeight)*crop.y/100,width:cropWidth,height:cropHeight,side:Math.max(cropWidth,cropHeight)};
}
export function cropImage(source:CanvasImageSource,width:number,height:number,crop:Crop):Promise<Blob>{
 const rect=cropRectangle(width,height,crop);const scale=Math.min(1,MAX_ROI_EDGE/Math.max(rect.width,rect.height));
 const outputWidth=Math.max(1,Math.floor(rect.width*scale));const outputHeight=Math.max(1,Math.floor(rect.height*scale));
 const canvas=document.createElement('canvas');canvas.width=outputWidth;canvas.height=outputHeight;
 const context=canvas.getContext('2d',{alpha:false});if(!context)throw new Error('Image canvas is unavailable.');
 context.fillStyle='rgb(128, 128, 128)';context.fillRect(0,0,outputWidth,outputHeight);
 context.imageSmoothingQuality='high';context.drawImage(source,rect.left,rect.top,rect.width,rect.height,0,0,outputWidth,outputHeight);
 return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('Could not prepare image.')),'image/png'));
}
