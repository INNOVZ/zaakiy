"""
Adaptive Web Scraper with Dynamic Concurrency Management
Implements queue-based task system with intelligent load balancing
"""

import asyncio
import statistics
import time

# Optional system monitoring (install with: pip install psutil)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from ...utils.logging_config import PerformanceLogger, get_logger
from .url_utils import create_safe_fetch_message
from .web_scraper import ScrapingConfig, SecureWebScraper

logger = get_logger(__name__)


@dataclass
class TaskMetrics:
    """Metrics for tracking task performance"""

    response_times: deque = field(default_factory=lambda: deque(maxlen=10))
    success_rate: float = 1.0
    error_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """System resource metrics"""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    queue_size: int = 0


@dataclass
class ScrapingTask:
    """Individual scraping task"""

    url: str
    depth: int
    priority: int = 1
    domain: str = field(init=False)
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3

    def __post_init__(self):
        self.domain = urlparse(self.url).netloc.lower()


class AdaptiveConcurrencyManager:
    """Manages dynamic concurrency based on performance metrics"""

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        target_response_time: float = 2.0,
        adjustment_interval: float = 5.0,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_response_time = target_response_time
        self.adjustment_interval = adjustment_interval

        self.current_workers = min_workers
        self.domain_metrics: Dict[str, TaskMetrics] = defaultdict(TaskMetrics)
        self.system_metrics = SystemMetrics()
        self.last_adjustment = time.time()

        self._worker_semaphores: Dict[str, asyncio.Semaphore] = {}

    def get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        """Get or create semaphore for domain with dynamic limits"""
        if domain not in self._worker_semaphores:
            # Calculate optimal concurrency for this domain
            optimal_workers = self._calculate_domain_concurrency(domain)
            self._worker_semaphores[domain] = asyncio.Semaphore(optimal_workers)
            logger.debug(
                f"Created semaphore for {domain} with {optimal_workers} workers"
            )

        return self._worker_semaphores[domain]

    def _calculate_domain_concurrency(self, domain: str) -> int:
        """Calculate optimal concurrency for a specific domain"""
        metrics = self.domain_metrics[domain]

        # Base concurrency on domain performance
        if metrics.total_requests < 5:
            # New domain - start conservative
            return max(1, self.current_workers // 2)

        avg_response = metrics.avg_response_time
        success_rate = metrics.success_rate

        # Performance-based adjustment
        if avg_response <= self.target_response_time and success_rate > 0.9:
            # Domain is fast and reliable - allow more concurrency
            return min(self.max_workers, self.current_workers + 1)
        elif avg_response > self.target_response_time * 2 or success_rate < 0.8:
            # Domain is slow or unreliable - reduce concurrency
            return max(1, self.current_workers // 2)
        else:
            # Moderate performance - use current level
            return self.current_workers

    def update_metrics(self, domain: str, response_time: float, success: bool):
        """Update performance metrics for domain"""
        metrics = self.domain_metrics[domain]

        metrics.response_times.append(response_time)
        metrics.total_requests += 1

        if success:
            metrics.success_rate = (metrics.success_rate * 0.9) + (1.0 * 0.1)
        else:
            metrics.error_count += 1
            metrics.success_rate = (metrics.success_rate * 0.9) + (0.0 * 0.1)

        # Calculate moving average
        if metrics.response_times:
            metrics.avg_response_time = statistics.mean(metrics.response_times)

        metrics.last_update = time.time()

        # Trigger adjustment if needed
        self._maybe_adjust_concurrency()

    def _maybe_adjust_concurrency(self):
        """Adjust global concurrency based on system performance"""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return

        self.last_adjustment = now
        self._update_system_metrics()

        # Calculate overall performance
        total_domains = len(self.domain_metrics)
        if total_domains == 0:
            return

        avg_response_time = statistics.mean(
            [
                m.avg_response_time
                for m in self.domain_metrics.values()
                if m.total_requests > 0
            ]
        )

        avg_success_rate = statistics.mean(
            [m.success_rate for m in self.domain_metrics.values()]
        )

        # System load consideration
        high_system_load = (
            self.system_metrics.cpu_usage > 80 or self.system_metrics.memory_usage > 85
        )

        # Adjustment logic
        old_workers = self.current_workers

        if high_system_load:
            # Reduce workers due to system load
            self.current_workers = max(self.min_workers, self.current_workers - 1)
        elif avg_response_time <= self.target_response_time and avg_success_rate > 0.9:
            # Performance is good - can increase
            self.current_workers = min(self.max_workers, self.current_workers + 1)
        elif (
            avg_response_time > self.target_response_time * 2 or avg_success_rate < 0.7
        ):
            # Performance is poor - reduce
            self.current_workers = max(self.min_workers, self.current_workers - 1)

        if old_workers != self.current_workers:
            logger.info(
                f"Adjusted global workers: {old_workers} -> {self.current_workers} "
                f"(avg_response: {avg_response_time:.2f}s, success: {avg_success_rate:.2%})"
            )

            # Update domain semaphores
            self._update_domain_semaphores()

    def _update_system_metrics(self):
        """Update system resource metrics"""
        if not PSUTIL_AVAILABLE:
            # Fallback to basic metrics when psutil is not available
            self.system_metrics.cpu_usage = 0.0
            self.system_metrics.memory_usage = 0.0
            return

        try:
            self.system_metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
            self.system_metrics.memory_usage = psutil.virtual_memory().percent
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            self.system_metrics.cpu_usage = 0.0
            self.system_metrics.memory_usage = 0.0

    def _update_domain_semaphores(self):
        """Update existing semaphores with new limits"""
        for domain, semaphore in self._worker_semaphores.items():
            new_limit = self._calculate_domain_concurrency(domain)

            # Create new semaphore with updated limit
            # Note: asyncio.Semaphore doesn't support limit changes, so we recreate
            self._worker_semaphores[domain] = asyncio.Semaphore(new_limit)


class PriorityTaskQueue:
    """Priority queue for scraping tasks with intelligent scheduling"""

    def __init__(self):
        self.queues: Dict[int, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        self.domain_queues: Dict[str, List[ScrapingTask]] = defaultdict(list)
        self._queue_lock = asyncio.Lock()

    async def put(self, task: ScrapingTask):
        """Add task to appropriate priority queue"""
        async with self._queue_lock:
            await self.queues[task.priority].put(task)
            self.domain_queues[task.domain].append(task)

    async def get(self) -> Optional[ScrapingTask]:
        """Get next task with priority and load balancing"""
        async with self._queue_lock:
            # Check priority queues (higher priority first)
            for priority in sorted(self.queues.keys(), reverse=True):
                queue = self.queues[priority]
                if not queue.empty():
                    task = await queue.get()
                    # Remove from domain queue
                    if task in self.domain_queues[task.domain]:
                        self.domain_queues[task.domain].remove(task)
                    return task

            return None

    def get_domain_queue_size(self, domain: str) -> int:
        """Get number of pending tasks for domain"""
        return len(self.domain_queues[domain])

    def get_total_size(self) -> int:
        """Get total number of pending tasks"""
        return sum(q.qsize() for q in self.queues.values())


class AdaptiveWebScraper(SecureWebScraper):
    """Enhanced web scraper with adaptive concurrency and intelligent task management"""

    def __init__(
        self,
        config: Optional[ScrapingConfig] = None,
        min_workers: int = 1,
        max_workers: int = 10,
    ):
        super().__init__(config)

        self.concurrency_manager = AdaptiveConcurrencyManager(
            min_workers=min_workers, max_workers=max_workers
        )
        self.task_queue = PriorityTaskQueue()
        self.active_tasks: Set[asyncio.Task] = set()
        self.results: Dict[str, str] = {}
        self._shutdown_event = asyncio.Event()

    async def scrape_urls_adaptive(
        self, urls: List[str], max_pages: Optional[int] = None
    ) -> Dict[str, str]:
        """Scrape multiple URLs with adaptive concurrency"""
        max_pages = max_pages or len(urls)

        logger.info(
            f"Starting adaptive scraping of {len(urls)} URLs with dynamic concurrency"
        )

        # Add tasks to queue
        for i, url in enumerate(urls[:max_pages]):
            task = ScrapingTask(url=url, depth=0, priority=1)
            await self.task_queue.put(task)

        # Start worker pool
        workers = []
        for i in range(self.concurrency_manager.min_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            workers.append(worker)

        # Wait for completion or early termination
        try:
            await self._wait_for_completion()
        finally:
            # Cleanup
            self._shutdown_event.set()
            await asyncio.gather(*workers, return_exceptions=True)

        logger.info(f"Adaptive scraping completed: {len(self.results)} pages scraped")
        return self.results.copy()

    async def _worker(self, worker_id: str):
        """Worker coroutine that processes tasks from the queue"""
        logger.debug(f"Worker {worker_id} started")

        while not self._shutdown_event.is_set():
            try:
                # Get next task
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                if task is None:
                    continue

                # Get domain-specific semaphore
                semaphore = self.concurrency_manager.get_domain_semaphore(task.domain)

                # Process task with concurrency control
                start_time = time.time()
                success = False

                async with semaphore:
                    try:
                        # Respect rate limiting
                        await self._respect_rate_limit(task.domain)

                        # Perform scraping
                        with PerformanceLogger(f"adaptive_scrape_{task.domain}"):
                            text = await self.scrape_url_text(task.url)
                            self.results[task.url] = text
                            success = True

                    except Exception as e:
                        task.attempts += 1
                        logger.warning(
                            f"Task failed (attempt {task.attempts}/{task.max_attempts}): "
                            f"{create_safe_fetch_message(task.url)}: {type(e).__name__}"
                        )

                        # Retry if not exceeded max attempts
                        if task.attempts < task.max_attempts:
                            task.priority = max(
                                1, task.priority - 1
                            )  # Lower priority for retries
                            await self.task_queue.put(task)

                # Update metrics
                response_time = time.time() - start_time
                self.concurrency_manager.update_metrics(
                    task.domain, response_time, success
                )

            except asyncio.TimeoutError:
                # No tasks available - continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

        logger.debug(f"Worker {worker_id} stopped")

    async def _wait_for_completion(self):
        """Wait for all tasks to complete"""
        while True:
            total_pending = self.task_queue.get_total_size()
            if total_pending == 0:
                # Wait a bit more to ensure all workers are done
                await asyncio.sleep(2)
                if self.task_queue.get_total_size() == 0:
                    break

            # Log progress
            completed = len(self.results)
            logger.info(f"Progress: {completed} completed, {total_pending} pending")

            await asyncio.sleep(5)  # Progress check interval

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            "global_workers": self.concurrency_manager.current_workers,
            "total_scraped": len(self.results),
            "pending_tasks": self.task_queue.get_total_size(),
            "domain_stats": {},
        }

        for domain, metrics in self.concurrency_manager.domain_metrics.items():
            if metrics.total_requests > 0:
                stats["domain_stats"][domain] = {
                    "avg_response_time": metrics.avg_response_time,
                    "success_rate": metrics.success_rate,
                    "total_requests": metrics.total_requests,
                    "current_workers": self.concurrency_manager._calculate_domain_concurrency(
                        domain
                    ),
                }

        return stats


# Convenience factory function
def create_adaptive_scraper(
    min_workers: int = 1, max_workers: int = 10
) -> AdaptiveWebScraper:
    """Create an adaptive scraper with optimal settings"""
    return AdaptiveWebScraper(min_workers=min_workers, max_workers=max_workers)
