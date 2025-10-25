#!/usr/bin/env python3
"""
Performance threshold checker for load test results

This script checks if load test results meet performance thresholds.
Used in CI/CD to fail builds if performance degrades.

Usage:
    python tests/load/check_thresholds.py reports/load_test_results_stats.csv

Exit codes:
    0 - All thresholds met
    1 - One or more thresholds exceeded
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Performance thresholds
THRESHOLDS = {
    "response_time_p50": 200,  # 50th percentile in ms
    "response_time_p95": 500,  # 95th percentile in ms
    "response_time_p99": 1000,  # 99th percentile in ms
    "error_rate": 1.0,  # Max error rate in percent
    "min_requests_per_second": 10,  # Minimum RPS
}


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def read_locust_stats(csv_path: Path) -> Dict[str, float]:
    """
    Read Locust stats CSV and extract key metrics

    Args:
        csv_path: Path to Locust stats CSV file

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Get the aggregated row (Type == "Aggregated")
                if row.get("Type") == "Aggregated" or row.get("Name") == "Aggregated":
                    metrics["total_requests"] = float(row.get("Request Count", 0))
                    metrics["failure_count"] = float(row.get("Failure Count", 0))
                    metrics["response_time_p50"] = float(
                        row.get("50%", row.get("Median Response Time", 0))
                    )
                    metrics["response_time_p95"] = float(
                        row.get("95%", row.get("95th percentile", 0))
                    )
                    metrics["response_time_p99"] = float(
                        row.get("99%", row.get("99th percentile", 0))
                    )
                    metrics["average_response_time"] = float(
                        row.get("Average Response Time", 0)
                    )

                    # Calculate RPS (requests per second)
                    total_time = float(row.get("Total Content Size", 1))  # Fallback
                    if metrics["total_requests"] > 0:
                        # Estimate if not directly available
                        metrics["requests_per_second"] = float(
                            row.get("Requests/s", metrics["total_requests"] / 60)
                        )

                    # Calculate error rate
                    if metrics["total_requests"] > 0:
                        metrics["error_rate"] = (
                            metrics["failure_count"] / metrics["total_requests"]
                        ) * 100
                    else:
                        metrics["error_rate"] = 0

                    break

    except FileNotFoundError:
        print(f"{Colors.RED}Error: Stats file not found: {csv_path}{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}Error reading stats file: {e}{Colors.END}")
        sys.exit(1)

    return metrics


def check_thresholds(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Check if metrics meet performance thresholds

    Args:
        metrics: Dictionary of performance metrics

    Returns:
        Tuple of (all_passed, list_of_failures)
    """
    failures = []

    # Check response time thresholds
    if metrics.get("response_time_p50", float("inf")) > THRESHOLDS["response_time_p50"]:
        failures.append(
            f"P50 response time: {metrics['response_time_p50']:.0f}ms "
            f"(threshold: {THRESHOLDS['response_time_p50']}ms)"
        )

    if metrics.get("response_time_p95", float("inf")) > THRESHOLDS["response_time_p95"]:
        failures.append(
            f"P95 response time: {metrics['response_time_p95']:.0f}ms "
            f"(threshold: {THRESHOLDS['response_time_p95']}ms)"
        )

    if metrics.get("response_time_p99", float("inf")) > THRESHOLDS["response_time_p99"]:
        failures.append(
            f"P99 response time: {metrics['response_time_p99']:.0f}ms "
            f"(threshold: {THRESHOLDS['response_time_p99']}ms)"
        )

    # Check error rate
    if metrics.get("error_rate", float("inf")) > THRESHOLDS["error_rate"]:
        failures.append(
            f"Error rate: {metrics['error_rate']:.2f}% "
            f"(threshold: {THRESHOLDS['error_rate']}%)"
        )

    # Check minimum RPS
    if metrics.get("requests_per_second", 0) < THRESHOLDS["min_requests_per_second"]:
        failures.append(
            f"Requests per second: {metrics['requests_per_second']:.2f} "
            f"(minimum: {THRESHOLDS['min_requests_per_second']})"
        )

    return len(failures) == 0, failures


def print_results(metrics: Dict[str, float], passed: bool, failures: List[str]):
    """
    Print formatted results

    Args:
        metrics: Performance metrics
        passed: Whether all thresholds passed
        failures: List of threshold failures
    """
    print()
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}Load Test Performance Results{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print()

    # Print metrics
    print(f"{Colors.BOLD}Performance Metrics:{Colors.END}")
    print(f"  Total Requests:     {metrics.get('total_requests', 0):.0f}")
    print(f"  Failure Count:      {metrics.get('failure_count', 0):.0f}")
    print(f"  Error Rate:         {metrics.get('error_rate', 0):.2f}%")
    print(f"  Requests/Second:    {metrics.get('requests_per_second', 0):.2f}")
    print(f"  Avg Response Time:  {metrics.get('average_response_time', 0):.0f}ms")
    print(f"  P50 Response Time:  {metrics.get('response_time_p50', 0):.0f}ms")
    print(f"  P95 Response Time:  {metrics.get('response_time_p95', 0):.0f}ms")
    print(f"  P99 Response Time:  {metrics.get('response_time_p99', 0):.0f}ms")
    print()

    # Print thresholds
    print(f"{Colors.BOLD}Performance Thresholds:{Colors.END}")
    print(f"  P50 Response Time:  <= {THRESHOLDS['response_time_p50']}ms")
    print(f"  P95 Response Time:  <= {THRESHOLDS['response_time_p95']}ms")
    print(f"  P99 Response Time:  <= {THRESHOLDS['response_time_p99']}ms")
    print(f"  Error Rate:         <= {THRESHOLDS['error_rate']}%")
    print(f"  Min RPS:            >= {THRESHOLDS['min_requests_per_second']}")
    print()

    # Print results
    if passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All thresholds passed!{Colors.END}")
        print()
    else:
        print(
            f"{Colors.RED}{Colors.BOLD}✗ Performance thresholds exceeded:{Colors.END}"
        )
        print()
        for failure in failures:
            print(f"  {Colors.RED}• {failure}{Colors.END}")
        print()

    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print()


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(f"{Colors.RED}Usage: {sys.argv[0]} <stats_csv_file>{Colors.END}")
        print(
            f"{Colors.YELLOW}Example: {sys.argv[0]} reports/load_test_results_stats.csv{Colors.END}"
        )
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    print(f"{Colors.BLUE}Reading stats from: {csv_path}{Colors.END}")

    # Read metrics
    metrics = read_locust_stats(csv_path)

    if not metrics:
        print(f"{Colors.RED}No metrics found in stats file{Colors.END}")
        sys.exit(1)

    # Check thresholds
    passed, failures = check_thresholds(metrics)

    # Print results
    print_results(metrics, passed, failures)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
