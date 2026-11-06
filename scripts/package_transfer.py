"""Build a local laptop-transfer ZIP with trained weights, excluding environments."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'node_modules', '.venv', '__pycache__', '.pytest_cache', 'dist', 'test-results', 'playwright-report'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-data', action='store_true', help='Include private original ZIP and source documents for your own laptop only')
    args = parser.parse_args()
    selected = set()
    for folder in ['backend', 'frontend', 'ml', 'scripts', 'docs', '.vscode', 'report',
                   'artifacts/checkpoints', 'artifacts/metrics', 'artifacts/plots']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and not EXCLUDED.intersection(path.relative_to(ROOT).parts) and path.suffix not in {'.log', '.pyc', '.tsbuildinfo'}:
                selected.add(path)
    for name in ['README.md', 'AGENTS.md', 'ARCHITECTURE.md', 'requirements.txt', 'requirements-gpu.txt', '.gitignore', '.env.example',
                 'artifacts/data_audit.json', 'artifacts/experiment_manifest.json', 'artifacts/model_architectures.json',
                 'artifacts/expansion_audit.json', 'artifacts/expansion_comparison.json', 'artifacts/expansion_comparison.csv']:
        path = ROOT / name
        if path.is_file():
            selected.add(path)
    if args.include_data:
        for name in ['data/ready_dataset.zip', 'source/Assignment2.docx', 'source/notebook.ipynb']:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            selected.add(path)
    for model in ['simple_cnn', 'resnet18', 'mobilenetv4']:
        if ROOT / f'artifacts/checkpoints/{model}.pt' not in selected:
            raise FileNotFoundError(f'Missing required trained model: {model}')
    output = ROOT / 'exports' / 'gesturelab-laptop.zip'
    output.parent.mkdir(exist_ok=True)
    records = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)}
    manifest = {'files': records, 'private_source_data_included': args.include_data,
                'privacy': 'Contains private image-bearing report. Transfer to your own laptop only; do not publish without verified rights.',
                'windows_start': ['powershell -ExecutionPolicy Bypass -File scripts/setup.ps1', 'powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1'],
                'notes': 'Install Python 3.10+ and Node.js first. Environments are rebuilt, trained weights are already included. No training/download of model weights required to serve. Dataset and demo video omitted unless explicitly included separately.'}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            archive.write(path, 'gesturelab/' + path.relative_to(ROOT).as_posix())
        archive.writestr('gesturelab/TRANSFER_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, expected in records.items():
            assert hashlib.sha256(archive.read('gesturelab/' + name)).hexdigest() == expected
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size, 'verified_files': len(records),
                      'includes_original_data': args.include_data, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
