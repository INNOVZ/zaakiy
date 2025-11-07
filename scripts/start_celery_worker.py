#!/usr/bin/env python
"""
Start Celery worker for background task processing.

Usage:
    python scripts/start_celery_worker.py

    # Or with specific queue
    python scripts/start_celery_worker.py --queue uploads

    # Or with concurrency
    python scripts/start_celery_worker.py --concurrency 4
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.celery_app import celery_app


def main():
    parser = argparse.ArgumentParser(description="Start Celery worker")
    parser.add_argument(
        "--queue",
        default="uploads",
        help="Queue name to consume from (default: uploads)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of worker processes (default: 2)",
    )
    parser.add_argument(
        "--loglevel",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Start worker
    celery_app.worker_main(
        [
            "worker",
            f"--loglevel={args.loglevel}",
            f"--concurrency={args.concurrency}",
            f"--queues={args.queue}",
            "--hostname=worker@%h",
        ]
    )


if __name__ == "__main__":
    main()
