"""Checkpoint-based integration evidence, no invented production predictions."""
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import create_app


def main():
    if all((ROOT/'artifacts/checkpoints'/f'{name}_finetuned.pt').is_file() for name in ('resnet18','mobilenetv4')):
        from scripts.smoke_expansion import main as expanded_main
        return expanded_main()
    output = ROOT / 'artifacts' / 'verification'
    output.mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app())
    health = client.get('/api/health').json()
    assert health['ready'], health
    evidence = {'health': health, 'predictions': [], 'checks': []}
    samples = [next((ROOT / 'data' / 'ready_dataset' / 'test' / label).glob('*.jpg'))
               for label in ('fist', 'peace')]
    for model in health['models']:
        results = []
        for image_path in samples:
            response = client.post('/api/predict', files={'file': ('heldout.jpg', image_path.read_bytes(), 'image/jpeg')},
                                   data={'model_id': model['id'], 'explain': 'true'})
            assert response.status_code == 200, response.text
            result = response.json()
            assert abs(sum(result['probabilities'].values()) - 1) < 1e-6
            cam = Image.open(io.BytesIO(base64.b64decode(result['explanation']['image_base64'])))
            assert cam.size == (224, 224) and np.asarray(cam).std() > 0
            if model['id'] == 'mobilenetv4' and len(results) == 0:
                # Local private report evidence; do not publish without consent.
                cam.save(output / 'gradcam_mobilenetv4.png')
                with Image.open(image_path) as original:
                    original.convert('RGB').save(output / 'input_mobilenetv4.png')
            del result['explanation']['image_base64']
            result['source'] = image_path.relative_to(ROOT).as_posix()
            evidence['predictions'].append(result)
            results.append(result)
        assert results[0]['probabilities'] != results[1]['probabilities']
        evidence['checks'].append(f"{model['id']}: strict checkpoint restore, two distinct held-out inputs, finite normalized probabilities, spatial Grad-CAM")
    experiments = client.get('/api/experiments').json()
    assert experiments['available'] and len(experiments['runs']) == 4
    evidence['checks'].append('Four measured experiment artifacts exposed by API')
    (output / 'integration.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'models': len(health['models']), 'predictions': len(evidence['predictions']),
                      'runs': len(experiments['runs']), 'evidence': str(output / 'integration.json')}, indent=2))


if __name__ == '__main__':
    main()
