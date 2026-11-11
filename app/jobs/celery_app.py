"""Celery application configured per ADR 003 (acks_late, prefetch 1, JSON only).

Attempt counting against the Redis-side job lifecycle (not Celery retries alone)
arrives with the real indexing task in stage 2+.
"""

from __future__ import annotations

from celery import Celery

from app.config.settings import Settings, get_settings


def create_celery_app(settings: Settings) -> Celery:
    celery_app = Celery(
        "rag_backend",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["app.jobs.tasks"],
    )
    celery_app.conf.update(
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        enable_utc=True,
        # Starting values per IMPLEMENTATION_PLAN.md §8.5; re-measured later.
        task_soft_time_limit=900,
        task_time_limit=1080,
        broker_transport_options={"visibility_timeout": 1200},
        result_expires=3600,
        worker_send_task_events=False,
        task_default_queue="indexing",
    )
    return celery_app


celery_app = create_celery_app(get_settings())
