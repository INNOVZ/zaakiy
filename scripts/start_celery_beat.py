#!/usr/bin/env python
"""
Start Celery Beat scheduler for periodic tasks.

Usage:
    python scripts/start_celery_beat.py
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.celery_app import celery_app


def main():
    """Start Celery Beat scheduler"""
    celery_app.start(
        [
            "beat",
            "--loglevel=info",
            "--scheduler=celery.beat:PersistentScheduler",
        ]
    )


if __name__ == "__main__":
    main()
