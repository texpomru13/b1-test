"""A stale server or an old export must stop an evaluation before generation."""
import json
from pathlib import Path

import httpx
import pytest
from app.domain.build import api_digest


def test_evaluator_rejects_a_different_running_build(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "eval"))
    from provenance import live_inputs

    def reply(request):
        assert request.url.path == "/health/build"
        return httpx.Response(200, json={"instance": "api", "source_sha256": "older-code"})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(ValueError, match="different source hashes"):
            live_inputs(client, None, {})


def test_evaluator_rejects_reindexed_document(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "eval"))
    import provenance

    monkeypatch.setattr(provenance, "WORK", tmp_path)
    path = tmp_path / "input.pdf"
    path.write_bytes(b"fixture")
    (tmp_path / "corpus.json").write_text(json.dumps({"docs": {"f": {"file": str(path)}}}))
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "f.json").write_text(json.dumps(
        {"accepted_run_id": "older-run", "index_fingerprint": "same-fingerprint"}))

    def reply(request):
        if request.url.path == "/health/build":
            return httpx.Response(200, json={"instance": "api", "source_sha256": api_digest()})
        assert request.url.path == "/api/v1/documents/doc"
        return httpx.Response(200, json={"ready": True, "accepted_run_id": "new-run",
                                         "index_fingerprint": "same-fingerprint",
                                         "source_sha256": provenance.sha256(path)})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(ValueError, match="differs from the live service"):
            provenance.live_inputs(client, None, {"f": "doc"})
