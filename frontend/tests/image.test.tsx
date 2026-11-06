import {afterEach,describe,expect,it,vi} from 'vitest';
import {cropImage,cropRectangle,MAX_ROI_EDGE} from '../src/image';

afterEach(()=>vi.restoreAllMocks());

describe('native hand-region preparation',()=>{
 it('keeps a tall hand rectangular instead of introducing camera background or stretching it',async()=>{
  const rect=cropRectangle(640,480,{x:50,y:50,size:75,aspect:.65});
  expect(rect.width).toBe(234);expect(rect.height).toBe(360);expect(rect.left).toBe(203);expect(rect.top).toBe(60);
  const context={fillStyle:'',fillRect:vi.fn(),drawImage:vi.fn(),imageSmoothingQuality:''};
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D);
  const outputs:HTMLCanvasElement[]=[];
  vi.spyOn(HTMLCanvasElement.prototype,'toBlob').mockImplementation(function(this:HTMLCanvasElement,callback){outputs.push(this);callback(new Blob(['test'],{type:'image/png'}));});
  await cropImage(document.createElement('img'),640,480,{x:50,y:50,size:75,aspect:.65});
  expect([outputs[0]?.width,outputs[0]?.height]).toEqual([234,360]);
  expect(context.drawImage.mock.calls[0].slice(1)).toEqual([203,60,234,360,0,0,234,360]);
 });
 it.each([[1200,800,75,600],[224,224,100,224],[256,256,100,256],[4000,3000,100,MAX_ROI_EDGE],[100,80,50,40]])('preserves crop detail within upload bounds for %ix%i at %i percent',async(width,height,size,expected)=>{
  const context={fillStyle:'',fillRect:vi.fn(),drawImage:vi.fn(),imageSmoothingQuality:''};
  const getContext=vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D);
  const canvases:HTMLCanvasElement[]=[];
  vi.spyOn(HTMLCanvasElement.prototype,'toBlob').mockImplementation(function(this:HTMLCanvasElement,callback){canvases.push(this);callback(new Blob(['image'],{type:'image/png'}));});
  const source=document.createElement('img');const blob=await cropImage(source,width,height,{x:50,y:50,size});
  expect(blob.type).toBe('image/png');expect(canvases[0].width).toBe(expected);expect(canvases[0].height).toBe(expected);
  expect(getContext).toHaveBeenCalledWith('2d',{alpha:false});
  expect(context.fillStyle).toBe('rgb(128, 128, 128)');expect(context.fillRect).toHaveBeenCalledWith(0,0,expected,expected);
  expect(context.fillRect.mock.invocationCallOrder[0]).toBeLessThan(context.drawImage.mock.invocationCallOrder[0]);
  expect(context.drawImage.mock.calls[0].slice(-4)).toEqual([0,0,expected,expected]);
 });
 it('rejects unavailable source dimensions',()=>{expect(()=>cropImage(document.createElement('img'),0,0,{x:50,y:50,size:100})).toThrow('dimensions must be positive');});
});
