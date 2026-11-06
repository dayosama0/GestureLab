export const CLASSES = ['fist', 'open_palm', 'peace', 'thumbs_up'] as const;
export type Gesture = typeof CLASSES[number];
export type Model = {id:string;name:string;available:boolean;reason?:string|null;version?:string|null;metadata?:Record<string,unknown>};
export type Preprocessing = {input_size?:number;channel_order?:string;mean?:number[];std?:number[];[key:string]:unknown};
export type Prediction = {model_id:string;version:string;label:Gesture;probabilities:Record<Gesture,number>;latency_ms:number;device:string;preprocessing:Preprocessing;explanation:null|{image_base64:string;input_image_base64?:string|null;mime_type:string;width:number;height:number;method:string}};
export type Metrics = {accuracy:number;precision_macro:number;recall_macro:number;f1_macro:number;per_class?:Record<string,unknown>};
export type TestEvaluation = {metrics:Metrics;confusion_matrix:number[][]};
export type Run = {run_id:string;model_id:string;metadata:Record<string,unknown>;history:{epoch:number;train_loss:number;val_loss:number;train_accuracy:number;val_accuracy:number}[];metrics:Metrics;confusion_matrix:number[][];latency?:{mean_ms:number;p50_ms:number;p95_ms:number;device:string};stress?:Record<string,Metrics>;original_test?:TestEvaluation;expanded_test?:TestEvaluation};
async function request<T>(url:string,options?:RequestInit):Promise<T>{
 const response=await fetch(url,options);
 if(!response.ok){let message=`Request failed (${response.status})`;try{const data=await response.json();message=typeof data.detail==='string'?data.detail:message;}catch{/* A non-JSON error still has an HTTP status. */}throw new Error(message);}
 return response.json();
}
export const getModels=()=>request<{classes:Gesture[];models:Model[]}>('/api/models');
export const getExperiments=()=>request<{available:boolean;runs:Run[]}>('/api/experiments');
export async function predict(file:Blob,modelId:string,explain=true):Promise<Prediction>{const body=new FormData();body.append('file',file,'hand-roi.png');body.append('model_id',modelId);body.append('explain',String(explain));return request('/api/predict',{method:'POST',body});}
export const label=(value:string)=>value.replaceAll('_',' ');
export const pct=(value:number)=>`${(value*100).toFixed(1)}%`;
export function inputDescription(preprocessing?:Preprocessing){const size=preprocessing?.input_size;if(typeof size!=='number'||!Number.isFinite(size)||size<=0)return null;const channel=preprocessing?.channel_order;return `${typeof channel==='string'?`${channel} · `:''}${size} × ${size} model input`;}
export function preferredModelId(models:Model[]):string{
 const ranked=models.flatMap(model=>{
  if(!model.available||!model.id.endsWith('_finetuned'))return [];
  const validation=model.metadata?.validation_metrics;
  if(!validation||typeof validation!=='object'||Array.isArray(validation))return [];
  const score=(validation as Record<string,unknown>).f1_macro;
  return typeof score==='number'&&Number.isFinite(score)&&score>=0&&score<=1?[{id:model.id,score}]:[];
 }).sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id));
 return ranked[0]?.id??models.find(model=>model.available)?.id??'';
}
