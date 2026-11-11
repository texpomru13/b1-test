from app.config.settings import Settings
from app.jobs.celery_app import create_celery_app


def test_celery_reliability_config() -> None:
    celery_app = create_celery_app(Settings(_env_file=None))  # type: ignore[call-arg]
    conf = celery_app.conf
    assert conf.task_acks_late is True
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_serializer == "json"
    assert conf.result_serializer == "json"
    assert conf.accept_content == ["json"]
    assert conf.broker_transport_options["visibility_timeout"] == 1200
    assert conf.task_time_limit > conf.task_soft_time_limit
    assert conf.broker_transport_options["visibility_timeout"] > conf.task_time_limit


def test_ping_task_registered() -> None:
    import app.jobs.tasks  # noqa: F401  (registration side effect)
    from app.jobs.celery_app import celery_app

    assert "app.jobs.ping" in celery_app.tasks
