"""The ClipEarth text encoder service, with a fake model (no network, no model files).

Parity with the lab's transformers.js encoding was checked by hand in Chromium (see the service README).
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")
pytest.importorskip("aiohttp")

from fastapi.testclient import TestClient

SERVICE = (
    Path(__file__).parents[3] / "olmoearth_pretrain" / "clipearth" / "encoder_service"
)
sys.path.insert(0, str(SERVICE))
import app as service
from encoder import DIMS, Encoder


class CountingEncoder(Encoder):
    """The real cache and batching around a fake model: row i is filled with len(text)."""

    def __init__(self):
        """No model: just the cache, the lock and a record of batches."""
        import threading
        from collections import OrderedDict

        self.cache, self.cache_size, self.lock = OrderedDict(), 2, threading.Lock()
        self.stats = {
            "requests": 0,
            "texts": 0,
            "cache_hits": 0,
            "runs": 0,
            "run_s": 0.0,
        }
        self.batches: list[list[str]] = []

    def _run(self, texts):
        self.batches.append(texts)
        return np.stack([np.full(DIMS, len(t), np.float32) for t in texts])


@pytest.fixture
def client(monkeypatch):
    """The app around a CountingEncoder, auth off."""
    encoder = CountingEncoder()
    monkeypatch.setitem(service.state, "encoder", encoder)
    monkeypatch.setattr(service, "NO_AUTH", True)
    with TestClient(service.app) as c:
        c.encoder = encoder
        yield c


def post(client, texts, model=service.MODEL, **kw):
    """POST /encode."""
    return client.post("/encode", json={"model": model, "texts": texts}, **kw)


def test_encode_returns_one_vector_per_text_in_order(client):
    """Rows come back in the order of the texts, repeats included."""
    r = post(client, ["rice", "wetland", "rice"])
    assert r.status_code == 200
    vectors = np.frombuffer(base64.b64decode(r.json()["vectors"]), "<f4").reshape(
        -1, DIMS
    )
    assert vectors[:, 0].tolist() == [4, 7, 4]


def test_repeated_texts_run_the_model_once(client):
    """Repeats within and across requests come from the cache."""
    post(client, ["rice", "wetland", "rice"])
    post(client, ["wetland"])
    assert client.encoder.batches == [["rice", "wetland"]]


def test_a_different_model_is_refused(client):
    """A head naming another text model gets 409."""
    assert post(client, ["rice"], model="some/other-model").status_code == 409


@pytest.mark.parametrize("texts", [[], [1], ["x"] * 65, ["x" * 1001]])
def test_bad_texts_are_refused(client, texts):
    """Empty, non-string, too many or too long texts get 400."""
    assert post(client, texts).status_code == 400


def test_encode_needs_a_key(client, monkeypatch):
    """Without a key the model never runs."""
    monkeypatch.setattr(service, "NO_AUTH", False)
    r = post(client, ["rice"])
    assert r.status_code == 401
    assert client.encoder.batches == []
