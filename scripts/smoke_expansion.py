"""Actual five-model, per-checkpoint preprocessing and explanation integration."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import create_app

def main():
    output=ROOT/'artifacts/verification'
    before=json.loads((output/'legacy_before_expansion.json').read_text())
    for path,digest in before.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    client=TestClient(create_app())
    health=client.get('/api/health').json()
    assert health['ready'] and len(health['models'])==5,health
    evidence={'health':health,'predictions':[],'legacy_artifacts_unchanged':True}
    samples=[next((ROOT/'data/ready_dataset/test'/label).glob('*.jpg')) for label in ['fist','peace']]
    for model in health['models']:
        size=model['metadata']['input_size'];probabilities=[]
        for index,sample in enumerate(samples):
            response=client.post('/api/predict',files={'file':('heldout.jpg',sample.read_bytes(),'image/jpeg')},
                                 data={'model_id':model['id'],'explain':'true'})
            assert response.status_code==200,(model['id'],response.text)
            result=response.json();assert abs(sum(result['probabilities'].values())-1)<1e-6
            assert result['preprocessing']['mean']==model['metadata']['mean']
            explanation=result['explanation']
            overlay=Image.open(io.BytesIO(base64.b64decode(explanation.pop('image_base64'))))
            prepared=Image.open(io.BytesIO(base64.b64decode(explanation.pop('input_image_base64'))))
            assert overlay.size==prepared.size==(size,size)
            assert np.asarray(overlay).std()>0
            if model['id']=='mobilenetv4_finetuned' and index==0:
                overlay.save(output/'gradcam_mobilenetv4_finetuned.png')
                prepared.save(output/'input_mobilenetv4_finetuned.png')
            result['source']=sample.relative_to(ROOT).as_posix();evidence['predictions'].append(result)
            probabilities.append(result['probabilities'])
        assert probabilities[0]!=probabilities[1]
    runs=client.get('/api/experiments').json()['runs']
    assert len(runs)==6
    for run in runs:
        if run['run_id'].endswith('_finetuned'):
            assert run['original_test']['metrics']['sample_count']==60
            assert run['expanded_test']['metrics']['sample_count']==400
    evidence['experiment_runs']=[r['run_id'] for r in runs]
    (output/'expansion_integration.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('PASS5 trained models,10 actual held-out predictions,6 measured runs,224/256 exact input/Grad-CAM alignment; legacy13 files unchanged')

if __name__=='__main__':main()
