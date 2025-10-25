#!/usr/bin/env python3
"""
Performance Analysis Tool for Load Test Results

Analyzes load test results and provides detailed insights.

Usage:
    python tests/load/analyze_performance.py reports/latest_stats.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def read_stats(csv_path: Path) -> List[Dict]:
    """Read Locust stats CSV"""
    results = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Type") != "":  # Skip aggregated row
                    results.append(row)
    except Exception as e:
        print(f"{Colors.RED}Error reading {csv_path}: {e}{Colors.END}")
    return results


def analyze_endpoint(row: Dict) -> Dict:
    """Analyze single endpoint performance"""
    total = int(row.get("Request Count", 0))
    failures = int(row.get("Failure Count", 0))
    success = total - failures

    return {
        "name": row.get("Name", "Unknown"),
        "type": row.get("Type", "GET"),
        "total": total,
        "success": success,
        "failures": failures,
        "success_rate": (success / total * 100) if total > 0 else 0,
        "median": float(row.get("Median Response Time", 0)),
        "avg": float(row.get("Average Response Time", 0)),
        "p95": float(row.get("95%", 0)),
        "p99": float(row.get("99%", 0)),
        "min": float(row.get("Min Response Time", 0)),
        "max": float(row.get("Max Response Time", 0)),
        "rps": float(row.get("Requests/s", 0)),
    }


def get_status_icon(success_rate: float, median: float) -> str:
    """Get status icon based on performance"""
    if success_rate >= 99 and median < 500:
        return f"{Colors.GREEN}✅ Excellent{Colors.END}"
    elif success_rate >= 95 and median < 1000:
        return f"{Colors.GREEN}✓ Good{Colors.END}"
    elif success_rate >= 80 and median < 2000:
        return f"{Colors.YELLOW}⚠ Acceptable{Colors.END}"
    elif success_rate >= 50:
        return f"{Colors.YELLOW}⚠ Poor{Colors.END}"
    else:
        return f"{Colors.RED}✗ Critical{Colors.END}"


def print_summary(endpoints: List[Dict]):
    """Print performance summary"""
    print()
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}Load Test Performance Analysis{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*80}{Colors.END}")
    print()

    # Overall statistics
    total_requests = sum(e["total"] for e in endpoints)
    total_success = sum(e["success"] for e in endpoints)
    total_failures = sum(e["failures"] for e in endpoints)
    overall_success_rate = (
        (total_success / total_requests * 100) if total_requests > 0 else 0
    )

    print(f"{Colors.BOLD}Overall Summary:{Colors.END}")
    print(f"  Total Requests:  {total_requests}")
    print(f"  Successful:      {total_success} ({overall_success_rate:.1f}%)")
    print(f"  Failed:          {total_failures} ({100-overall_success_rate:.1f}%)")
    print()

    # Endpoint breakdown
    print(f"{Colors.BOLD}Endpoint Performance:{Colors.END}")
    print()
    print(
        f"{'Endpoint':<35} {'Requests':>8} {'Success':>7} {'Median':>8} {'P95':>8} {'Status':<20}"
    )
    print("-" * 100)

    for ep in sorted(endpoints, key=lambda x: x["success_rate"], reverse=True):
        status = get_status_icon(ep["success_rate"], ep["median"])
        print(
            f"{ep['name'][:34]:<35} "
            f"{ep['total']:>8} "
            f"{ep['success_rate']:>6.1f}% "
            f"{ep['median']:>7.0f}ms "
            f"{ep['p95']:>7.0f}ms "
            f"{status}"
        )

    print()

    # Performance categories
    print(f"{Colors.BOLD}Performance Breakdown:{Colors.END}")
    print()

    fast = [e for e in endpoints if e["median"] < 500]
    medium = [e for e in endpoints if 500 <= e["median"] < 2000]
    slow = [e for e in endpoints if 2000 <= e["median"] < 5000]
    very_slow = [e for e in endpoints if e["median"] >= 5000]

    print(f"  {Colors.GREEN}Fast (<500ms):{Colors.END}     {len(fast)} endpoints")
    for e in fast:
        print(f"    • {e['name']} ({e['median']:.0f}ms)")

    if medium:
        print(
            f"  {Colors.YELLOW}Medium (500-2000ms):{Colors.END} {len(medium)} endpoints"
        )
        for e in medium:
            print(f"    • {e['name']} ({e['median']:.0f}ms)")

    if slow:
        print(f"  {Colors.YELLOW}Slow (2-5s):{Colors.END}      {len(slow)} endpoints")
        for e in slow:
            print(f"    • {e['name']} ({e['median']:.0f}ms)")

    if very_slow:
        print(f"  {Colors.RED}Very Slow (>5s):{Colors.END}  {len(very_slow)} endpoints")
        for e in very_slow:
            print(f"    • {e['name']} ({e['median']:.0f}ms) ⚠️")

    print()

    # Recommendations
    print(f"{Colors.BOLD}Recommendations:{Colors.END}")
    print()

    if very_slow:
        print(f"  {Colors.RED}🚨 Critical:{Colors.END}")
        print(f"    • {len(very_slow)} endpoint(s) taking >5 seconds")
        print(f"    • Check: OpenAI API latency, Pinecone queries, database")
        print(f"    • Consider: Caching, async processing, streaming responses")
        print()

    failing = [e for e in endpoints if e["success_rate"] < 50]
    if failing:
        print(f"  {Colors.RED}🚨 High Failure Rate:{Colors.END}")
        for e in failing:
            print(f"    • {e['name']} - {e['success_rate']:.1f}% success rate")
        print(f"    • Check error logs for these endpoints")
        print()

    health_endpoint = next(
        (e for e in endpoints if "health" in e["name"].lower()), None
    )
    if health_endpoint and health_endpoint["median"] > 100:
        print(f"  {Colors.YELLOW}⚠️ Health Endpoint Slow:{Colors.END}")
        print(
            f"    • /health taking {health_endpoint['median']:.0f}ms (should be <100ms)"
        )
        print(f"    • Indicates: Database connection issues or app state problems")
        print()

    chat_endpoints = [
        e for e in endpoints if "chat" in e["name"].lower() and e["success_rate"] > 50
    ]
    if chat_endpoints:
        avg_chat_time = sum(e["median"] for e in chat_endpoints) / len(chat_endpoints)
        if avg_chat_time > 3000:
            print(f"  {Colors.YELLOW}💡 AI Performance:{Colors.END}")
            print(f"    • Average chat response: {avg_chat_time:.0f}ms")
            print(f"    • Consider: Response streaming, caching, faster models")
            print()

    # Success stories
    excellent = [e for e in endpoints if e["success_rate"] >= 99 and e["median"] < 500]
    if excellent:
        print(f"  {Colors.GREEN}✅ Excellent Performance:{Colors.END}")
        for e in excellent:
            print(
                f"    • {e['name']} - {e['median']:.0f}ms, {e['success_rate']:.1f}% success"
            )
        print()

    print(f"{Colors.BLUE}{Colors.BOLD}{'='*80}{Colors.END}")
    print()


def main():
    if len(sys.argv) < 2:
        print(f"{Colors.RED}Usage: {sys.argv[0]} <stats_csv_file>{Colors.END}")
        print(
            f"{Colors.YELLOW}Example: {sys.argv[0]} reports/quick_test_*_stats.csv{Colors.END}"
        )
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"{Colors.RED}File not found: {csv_path}{Colors.END}")
        sys.exit(1)

    print(f"{Colors.CYAN}Analyzing: {csv_path.name}{Colors.END}")

    rows = read_stats(csv_path)

    if not rows:
        print(f"{Colors.RED}No data found in file{Colors.END}")
        sys.exit(1)

    endpoints = [analyze_endpoint(row) for row in rows]
    print_summary(endpoints)


if __name__ == "__main__":
    main()
