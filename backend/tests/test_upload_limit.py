"""Privacy and bounded-receive tests independent of trained model availability."""
import asyncio
import io

import httpx
from PIL import Image
from fastapi.testclient import TestClient
import starlette.formparsers

from backend.app.main import create_app
from backend.app.upload_limit import MAX_REQUEST_BYTES


def test_chunked_body_rejected_before_multipart_parsing(tmp_path, monkeypatch):
    calls = []
    original = starlette.formparsers.SpooledTemporaryFile

    def track_spool(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(starlette.formparsers, 'SpooledTemporaryFile', track_spool)
    app = create_app(tmp_path / 'checkpoints', tmp_path / 'metrics')

    async def request():
        async def body():
            for _ in range(12):
                yield b'x' * (1024 * 1024)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            return await client.post('/api/predict', content=body(), headers={'Content-Type': 'multipart/form-data; boundary=qa'})

    response = asyncio.run(request())
    assert response.status_code == 413
    assert response.headers['cache-control'] == 'no-store'
    assert calls == [], 'Oversized body reached multipart parsing'


def test_accepted_large_image_never_spools_to_disk(tmp_path, monkeypatch):
    # A valid PNG >1MB (Starlette default disk-spool threshold).
    import numpy as np
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (900, 900, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    payload = buffer.getvalue()
    assert 1024 * 1024 < len(payload) < MAX_REQUEST_BYTES
    spools = []
    original = starlette.formparsers.SpooledTemporaryFile

    def track_spool(*args, **kwargs):
        spool = original(*args, **kwargs)
        spools.append(spool)
        return spool

    monkeypatch.setattr(starlette.formparsers, 'SpooledTemporaryFile', track_spool)
    client = TestClient(create_app(tmp_path / 'checkpoints', tmp_path / 'metrics'))
    response = client.post('/api/predict', files={'file': ('private.png', payload, 'image/png')}, data={'model_id': 'simple_cnn'})
    assert response.status_code == 503  # No checkpoint, but actual image parsed.
    assert len(spools) == 1
    assert spools[0]._rolled is False
    assert spools[0].closed
