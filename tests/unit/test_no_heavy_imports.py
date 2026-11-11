"""The API/worker package must never import model weights (single-owner rule, S8)."""

import inspect
import sys

HEAVY_MODULES = ("torch", "transformers", "sentence_transformers")


def test_app_does_not_import_heavy_ml_stacks() -> None:
    import app.inference.client
    import app.jobs.tasks
    import app.main  # noqa: F401

    loaded = [name for name in HEAVY_MODULES if name in sys.modules]
    assert loaded == [], f"heavy ML modules must not be imported by the app: {loaded}"


def test_health_endpoints_are_async() -> None:
    from app.api.health import live, ready

    assert inspect.iscoroutinefunction(live)
    assert inspect.iscoroutinefunction(ready)
