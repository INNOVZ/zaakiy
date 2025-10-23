#!/usr/bin/env python3
"""
Redis Cache Monitoring Script
Real-time monitoring of cache performance and health
"""
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.env_loader import safe_load_dotenv

# Load environment variables
safe_load_dotenv()


class Colors:
    """ANSI color codes"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def format_number(num: int) -> str:
    """Format number with thousand separators"""
    return f"{num:,}"


async def monitor_cache(interval: int = 5, iterations: int = None):
    """
    Monitor Redis cache performance

    Args:
        interval: Refresh interval in seconds
        iterations: Number of iterations (None = infinite)
    """
    try:
        import redis
    except ImportError:
        print(f"{Colors.RED}❌ Redis client not installed{Colors.END}")
        print(f"Run: pip install redis")
        sys.exit(1)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD")

    try:
        r = redis.from_url(
            redis_url,
            password=redis_password if redis_password else None,
            decode_responses=True,
        )
        r.ping()
    except redis.ConnectionError:
        print(f"{Colors.RED}❌ Cannot connect to Redis{Colors.END}")
        print(f"URL: {redis_url}")
        print(f"Is Redis running? Check with: redis-cli ping")
        sys.exit(1)

    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}🔍 Redis Cache Monitor{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

    iteration = 0
    last_hits = 0
    last_misses = 0
    last_time = time.time()

    try:
        while iterations is None or iteration < iterations:
            # Clear screen (optional)
            # os.system('clear' if os.name != 'nt' else 'cls')

            # Get current stats
            info_stats = r.info("stats")
            info_memory = r.info("memory")
            info_server = r.info("server")

            current_hits = info_stats.get("keyspace_hits", 0)
            current_misses = info_stats.get("keyspace_misses", 0)
            current_time = time.time()

            # Calculate rates
            time_diff = current_time - last_time
            hits_per_sec = (
                (current_hits - last_hits) / time_diff if time_diff > 0 else 0
            )
            misses_per_sec = (
                (current_misses - last_misses) / time_diff if time_diff > 0 else 0
            )

            # Calculate hit rate
            total_requests = current_hits + current_misses
            hit_rate = (
                (current_hits / total_requests * 100) if total_requests > 0 else 0
            )

            # Display timestamp
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{Colors.CYAN}⏰ {now}{Colors.END} (Refresh: {interval}s)\n")

            # Server info
            print(f"{Colors.BOLD}📊 Server Info:{Colors.END}")
            print(f"  Version:  {info_server.get('redis_version', 'unknown')}")
            print(f"  Uptime:   {info_server.get('uptime_in_seconds', 0)} seconds")
            print(f"  Port:     {info_server.get('tcp_port', 'unknown')}")

            # Cache statistics
            print(f"\n{Colors.BOLD}📈 Cache Statistics:{Colors.END}")
            print(
                f"  Total Hits:       {Colors.GREEN}{format_number(current_hits)}{Colors.END}"
            )
            print(
                f"  Total Misses:     {Colors.YELLOW}{format_number(current_misses)}{Colors.END}"
            )
            print(f"  Total Requests:   {format_number(total_requests)}")

            # Hit rate with color coding
            if hit_rate >= 80:
                color = Colors.GREEN
                status = "Excellent 🔥"
            elif hit_rate >= 60:
                color = Colors.YELLOW
                status = "Good ✅"
            else:
                color = Colors.RED
                status = "Needs Improvement ⚠️"

            print(f"  Hit Rate:         {color}{hit_rate:.2f}%{Colors.END} ({status})")

            # Real-time rates
            print(f"\n{Colors.BOLD}⚡ Real-Time Rates:{Colors.END}")
            print(f"  Hits/sec:         {hits_per_sec:.2f}")
            print(f"  Misses/sec:       {misses_per_sec:.2f}")
            print(f"  Requests/sec:     {hits_per_sec + misses_per_sec:.2f}")

            # Memory usage
            print(f"\n{Colors.BOLD}💾 Memory Usage:{Colors.END}")
            used_memory = info_memory.get("used_memory", 0)
            max_memory = info_memory.get("maxmemory", 0)

            print(f"  Used:             {format_bytes(used_memory)}")
            if max_memory > 0:
                memory_pct = used_memory / max_memory * 100
                print(f"  Max:              {format_bytes(max_memory)}")

                if memory_pct >= 90:
                    color = Colors.RED
                elif memory_pct >= 70:
                    color = Colors.YELLOW
                else:
                    color = Colors.GREEN

                print(f"  Usage:            {color}{memory_pct:.2f}%{Colors.END}")
            else:
                print(f"  Max:              {Colors.YELLOW}Not set{Colors.END}")

            print(
                f"  Fragmentation:    {info_memory.get('mem_fragmentation_ratio', 'N/A')}"
            )

            # Key information
            print(f"\n{Colors.BOLD}🔑 Keys:{Colors.END}")
            total_keys = 0
            for db_name, db_info in r.info("keyspace").items():
                keys_count = db_info.get("keys", 0)
                expires_count = db_info.get("expires", 0)
                total_keys += keys_count
                print(
                    f"  {db_name}:  {format_number(keys_count)} keys ({format_number(expires_count)} with TTL)"
                )

            if total_keys == 0:
                print(f"  {Colors.YELLOW}No keys in database{Colors.END}")

            # Count scraping-specific keys
            scraping_keys = 0
            try:
                for key in r.scan_iter(match="scrape:*", count=100):
                    scraping_keys += 1
                    if scraping_keys >= 1000:  # Limit scan
                        break
                if scraping_keys > 0:
                    print(f"\n{Colors.BOLD}🕷️  Scraping Cache:{Colors.END}")
                    print(f"  Active entries:   {format_number(scraping_keys)}")
            except Exception:
                pass

            # Eviction info
            evicted_keys = info_stats.get("evicted_keys", 0)
            if evicted_keys > 0:
                print(f"\n{Colors.BOLD}🗑️  Evictions:{Colors.END}")
                print(
                    f"  Total evicted:    {Colors.YELLOW}{format_number(evicted_keys)}{Colors.END}"
                )

            # Recommendations
            print(f"\n{Colors.BOLD}💡 Recommendations:{Colors.END}")
            if hit_rate < 50:
                print(f"  {Colors.RED}• Hit rate is low (<50%). Consider:")
                print(f"    - Increasing cache TTL")
                print(f"    - Enabling URL normalization")
                print(f"    - Reviewing cache invalidation logic{Colors.END}")
            elif hit_rate < 70:
                print(
                    f"  {Colors.YELLOW}• Hit rate is moderate. Can be improved.{Colors.END}"
                )
            else:
                print(f"  {Colors.GREEN}• Cache is performing well! 🎉{Colors.END}")

            if max_memory == 0:
                print(
                    f"  {Colors.YELLOW}• No max memory set. Consider setting a limit.{Colors.END}"
                )
            elif memory_pct > 85:
                print(
                    f"  {Colors.RED}• Memory usage is high. Consider increasing max memory.{Colors.END}"
                )

            print(f"\n{Colors.BOLD}{'-'*80}{Colors.END}")

            # Update last values
            last_hits = current_hits
            last_misses = current_misses
            last_time = current_time

            iteration += 1

            if iterations is None or iteration < iterations:
                await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}👋 Monitoring stopped{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}\n")
        raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor Redis cache performance in real-time"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations (default: infinite)",
    )

    args = parser.parse_args()

    asyncio.run(monitor_cache(interval=args.interval, iterations=args.iterations))


if __name__ == "__main__":
    main()
