"""One final same-holdout comparison; immutable legacy artifacts stay unchanged."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ml.finetune import ManifestDataset, validate_manifest
from ml.evaluate import evaluate, benchmark
from ml.models import MODEL_IDS, FINETUNED_IDS, create_model


def main():
    output = ROOT / 'artifacts' / 'expansion_comparison.json'
    if output.exists():
        raise FileExistsError('Comparison already exists; preserve this final evaluation')
    manifest_path = ROOT / 'data' / 'expanded' / 'manifest.json'
    _, groups, audit = validate_manifest(manifest_path)
    torch.set_num_threads(2)
    baseline_snapshot = json.loads((ROOT/'artifacts/verification/legacy_before_expansion.json').read_text())
    for path, expected in baseline_snapshot.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    rows = []
    for model_id in (*MODEL_IDS, *FINETUNED_IDS):
        checkpoint = ROOT / 'artifacts' / 'checkpoints' / f'{model_id}.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        metadata = payload['metadata']
        assert metadata['trained'] is True and metadata['model_id'] == model_id
        policy = {key: metadata[key] for key in ('input_size', 'mean', 'std')}
        dataset = ManifestDataset(groups['test'], policy)
        model = create_model(model_id, pretrained=False).eval()
        model.load_state_dict(payload['state_dict'], strict=True)
        if model_id in FINETUNED_IDS:
            # Reuse the already executed final test, without another model-selection pass.
            result = json.loads((ROOT/'artifacts/metrics'/f'{model_id}.json').read_text())
            assert result['metadata']['data_audit']['manifest_sha256'] == audit['manifest_sha256']
            metrics = result['expanded_test']['metrics']
            matrix = result['expanded_test']['confusion_matrix']
            source = f'artifacts/metrics/{model_id}.json#expanded_test'
        else:
            metrics, matrix, predictions = evaluate(model, dataset, batch_size=16)
            source = f'artifacts/metrics/predictions_{model_id}_expanded_test.csv'
            with (ROOT/source).open('x', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['source_id','user_id','actual','predicted'])
                writer.writeheader()
                labels = metadata['classes']
                writer.writerows({'source_id': record['source_id'], 'user_id':record['user_id'],
                                  'actual':record['label'], 'predicted':labels[index]}
                                 for record,index in zip(groups['test'],predictions))
        assert metrics['sample_count'] == len(groups['test'])
        timing = benchmark(model, dataset[0][0])
        rows.append({'model_id':model_id, 'metrics':metrics, 'confusion_matrix':matrix,
                     'latency_same_host_cpu':timing, 'prediction_source':source,
                     'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     'preprocessing':policy})
        print(f"{model_id}: expanded-test accuracy={metrics['accuracy']:.4f} F1={metrics['f1_macro']:.4f} CPU={timing['mean_ms']:.2f}ms", flush=True)
        del model,dataset
    result = {'manifest_sha256':audit['manifest_sha256'], 'expanded_test_count':len(groups['test']),
              'selection_policy':'New validation only. These final test results do not choose the default model.',
              'caveat':'Expansion and backbone fine-tuning both changed; this comparison does not isolate either factor.',
              'legacy_artifacts_unchanged':True, 'models':rows}
    with output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2)
    with output.with_suffix('.csv').open('x',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['model_id','expanded_test_accuracy','expanded_test_f1_macro','cpu_mean_ms'])
        writer.writeheader()
        writer.writerows({'model_id':r['model_id'],'expanded_test_accuracy':r['metrics']['accuracy'],
                          'expanded_test_f1_macro':r['metrics']['f1_macro'],
                          'cpu_mean_ms':r['latency_same_host_cpu']['mean_ms']} for r in rows)
    print('PASS all five models compared on the same new holdout:',output)


if __name__=='__main__':
    main()
