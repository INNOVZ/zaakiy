#!/usr/bin/env python3
"""
Load test results comparison tool

Compare performance metrics across multiple test runs to detect regressions.

Usage:
    python tests/load/compare_results.py baseline.csv current.csv
    python tests/load/compare_results.py reports/*.csv  # Compare all reports

Examples:
    # Compare two specific runs
    python tests/load/compare_results.py \
        reports/baseline_stats.csv \
        reports/latest_stats.csv

    # Compare all runs (shows trend)
    python tests/load/compare_results.py reports/*_stats.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def read_stats(csv_path: Path) -> Dict[str, float]:
    """Read Locust stats CSV and extract metrics"""
    metrics = {"file": csv_path.name}

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Type") == "Aggregated" or row.get("Name") == "Aggregated":
                    metrics["requests"] = float(row.get("Request Count", 0))
                    metrics["failures"] = float(row.get("Failure Count", 0))
                    metrics["p50"] = float(
                        row.get("50%", row.get("Median Response Time", 0))
                    )
                    metrics["p95"] = float(
                        row.get("95%", row.get("95th percentile", 0))
                    )
                    metrics["p99"] = float(
                        row.get("99%", row.get("99th percentile", 0))
                    )
                    metrics["avg"] = float(row.get("Average Response Time", 0))
                    metrics["rps"] = float(row.get("Requests/s", 0))

                    if metrics["requests"] > 0:
                        metrics["error_rate"] = (
                            metrics["failures"] / metrics["requests"]
                        ) * 100
                    else:
                        metrics["error_rate"] = 0
                    break
    except Exception as e:
        print(f"{Colors.RED}Error reading {csv_path}: {e}{Colors.END}")
        return {}

    return metrics


def calculate_diff(baseline: float, current: float) -> tuple[float, str, str]:
    """
    Calculate percentage difference and return color-coded string

    Returns:
        (percentage_diff, color, symbol)
    """
    if baseline == 0:
        return 0, Colors.YELLOW, "="

    diff = ((current - baseline) / baseline) * 100

    # For response times and error rates, higher is worse
    # For RPS and requests, higher is better
    if abs(diff) < 5:  # Less than 5% change
        color = Colors.GREEN
        symbol = "≈"
    elif diff > 0:
        color = Colors.RED  # Worse
        symbol = "↑"
    else:
        color = Colors.GREEN  # Better
        symbol = "↓"

    return diff, color, symbol


def compare_two_runs(baseline: Dict, current: Dict):
    """Compare two test runs"""
    print()
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}Load Test Comparison{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print()

    print(f"{Colors.BOLD}Baseline:{Colors.END} {baseline['file']}")
    print(f"{Colors.BOLD}Current: {Colors.END} {current['file']}")
    print()

    # Response times comparison (lower is better)
    metrics_lower_is_better = [
        ("P50 Response", "p50", "ms"),
        ("P95 Response", "p95", "ms"),
        ("P99 Response", "p99", "ms"),
        ("Avg Response", "avg", "ms"),
        ("Error Rate", "error_rate", "%"),
    ]

    metrics_higher_is_better = [
        ("Requests/sec", "rps", "RPS"),
        ("Total Requests", "requests", ""),
    ]

    print(f"{Colors.BOLD}Response Times & Error Rate (Lower is Better):{Colors.END}")
    print(f"{'Metric':<20} {'Baseline':>12} {'Current':>12} {'Change':>15}")
    print("-" * 70)

    for name, key, unit in metrics_lower_is_better:
        baseline_val = baseline.get(key, 0)
        current_val = current.get(key, 0)
        diff, _, _ = calculate_diff(baseline_val, current_val)

        # Invert color logic for "lower is better" metrics
        if abs(diff) < 5:
            color = Colors.GREEN
            symbol = "≈"
        elif diff > 0:
            color = Colors.RED
            symbol = "↑"
        else:
            color = Colors.GREEN
            symbol = "↓"

        change_str = f"{color}{symbol} {abs(diff):>6.1f}%{Colors.END}"

        if unit:
            print(
                f"{name:<20} {baseline_val:>10.1f}{unit:>2} "
                f"{current_val:>10.1f}{unit:>2} {change_str}"
            )
        else:
            print(
                f"{name:<20} {baseline_val:>12.0f} "
                f"{current_val:>12.0f} {change_str}"
            )

    print()
    print(f"{Colors.BOLD}Throughput (Higher is Better):{Colors.END}")
    print(f"{'Metric':<20} {'Baseline':>12} {'Current':>12} {'Change':>15}")
    print("-" * 70)

    for name, key, unit in metrics_higher_is_better:
        baseline_val = baseline.get(key, 0)
        current_val = current.get(key, 0)
        diff, _, _ = calculate_diff(baseline_val, current_val)

        # Normal color logic for "higher is better" metrics
        if abs(diff) < 5:
            color = Colors.GREEN
            symbol = "≈"
        elif diff > 0:
            color = Colors.GREEN
            symbol = "↑"
        else:
            color = Colors.RED
            symbol = "↓"

        change_str = f"{color}{symbol} {abs(diff):>6.1f}%{Colors.END}"

        if unit:
            print(
                f"{name:<20} {baseline_val:>10.1f} {unit:>2} "
                f"{current_val:>10.1f} {unit:>2} {change_str}"
            )
        else:
            print(
                f"{name:<20} {baseline_val:>12.0f} "
                f"{current_val:>12.0f} {change_str}"
            )

    print()

    # Summary
    p95_diff = calculate_diff(baseline.get("p95", 0), current.get("p95", 0))[0]
    error_diff = calculate_diff(
        baseline.get("error_rate", 0), current.get("error_rate", 0)
    )[0]

    print(f"{Colors.BOLD}Summary:{Colors.END}")

    if p95_diff < -10:
        print(
            f"  {Colors.GREEN}✓ Performance improved significantly (P95 down {abs(p95_diff):.1f}%){Colors.END}"
        )
    elif p95_diff > 10:
        print(
            f"  {Colors.RED}⚠ Performance degraded (P95 up {p95_diff:.1f}%){Colors.END}"
        )
    else:
        print(f"  {Colors.GREEN}✓ Performance is stable{Colors.END}")

    if error_diff > 10:
        print(f"  {Colors.RED}⚠ Error rate increased by {error_diff:.1f}%{Colors.END}")
    else:
        print(f"  {Colors.GREEN}✓ Error rate is acceptable{Colors.END}")

    print()
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print()


def show_trend(all_metrics: List[Dict]):
    """Show trend across multiple test runs"""
    print()
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}Performance Trend Analysis{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print()

    print(f"Analyzing {len(all_metrics)} test runs:")
    print()

    # Show table
    print(f"{'File':<30} {'P95 (ms)':>10} {'Error %':>10} {'RPS':>10}")
    print("-" * 70)

    for metrics in all_metrics:
        print(
            f"{metrics['file']:<30} "
            f"{metrics.get('p95', 0):>10.1f} "
            f"{metrics.get('error_rate', 0):>10.2f} "
            f"{metrics.get('rps', 0):>10.1f}"
        )

    print()

    # Calculate trend
    if len(all_metrics) >= 2:
        first = all_metrics[0]
        last = all_metrics[-1]

        p95_trend = calculate_diff(first.get("p95", 0), last.get("p95", 0))[0]
        error_trend = calculate_diff(
            first.get("error_rate", 0), last.get("error_rate", 0)
        )[0]

        print(f"{Colors.BOLD}Trend (First → Last):{Colors.END}")

        if p95_trend < -10:
            print(
                f"  {Colors.GREEN}✓ Response time improved by {abs(p95_trend):.1f}%{Colors.END}"
            )
        elif p95_trend > 10:
            print(
                f"  {Colors.RED}⚠ Response time degraded by {p95_trend:.1f}%{Colors.END}"
            )
        else:
            print(f"  {Colors.GREEN}≈ Response time stable{Colors.END}")

        if error_trend > 10:
            print(
                f"  {Colors.RED}⚠ Error rate increased by {error_trend:.1f}%{Colors.END}"
            )
        else:
            print(f"  {Colors.GREEN}✓ Error rate acceptable{Colors.END}")

    print()
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print()


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(
            f"{Colors.RED}Usage: {sys.argv[0]} <stats_csv_file1> [stats_csv_file2] ...{Colors.END}"
        )
        print()
        print(f"{Colors.YELLOW}Examples:{Colors.END}")
        print(f"  {sys.argv[0]} reports/baseline_stats.csv reports/latest_stats.csv")
        print(f"  {sys.argv[0]} reports/*_stats.csv")
        sys.exit(1)

    # Read all provided files
    all_metrics = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists():
            metrics = read_stats(path)
            if metrics:
                all_metrics.append(metrics)

    if len(all_metrics) == 0:
        print(f"{Colors.RED}No valid stats files found{Colors.END}")
        sys.exit(1)

    if len(all_metrics) == 1:
        print(
            f"{Colors.YELLOW}Only one file provided. Need at least 2 for comparison.{Colors.END}"
        )
        metrics = all_metrics[0]
        print()
        print(f"{Colors.BOLD}Metrics from {metrics['file']}:{Colors.END}")
        print(f"  P50 Response: {metrics.get('p50', 0):.1f}ms")
        print(f"  P95 Response: {metrics.get('p95', 0):.1f}ms")
        print(f"  P99 Response: {metrics.get('p99', 0):.1f}ms")
        print(f"  Error Rate:   {metrics.get('error_rate', 0):.2f}%")
        print(f"  RPS:          {metrics.get('rps', 0):.1f}")
        sys.exit(0)

    if len(all_metrics) == 2:
        # Compare two runs
        compare_two_runs(all_metrics[0], all_metrics[1])
    else:
        # Show trend across all runs
        show_trend(all_metrics)


if __name__ == "__main__":
    main()
