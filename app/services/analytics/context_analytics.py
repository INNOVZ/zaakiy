import os
import logging
import statistics
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from dataclasses import dataclass
from ..storage.supabase_client import get_supabase_client
# import json


@dataclass
class ContextMetrics:
    """Context engineering metrics data class"""
    org_id: str
    conversation_id: str
    message_id: str
    query_original: str
    query_enhanced: List[str]
    documents_retrieved: List[Dict]
    context_length: int
    context_quality_score: float
    retrieval_time_ms: int
    response_time_ms: int
    model_used: str
    sources_count: int
    user_satisfaction: Optional[float] = None
    feedback_text: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ContextAnalytics:
    """Advanced analytics for context engineering performance"""

    def __init__(self):
        try:
            self.supabase = get_supabase_client()
        except Exception as e:
            logging.warning(
                f"Failed to initialize Supabase client. Analytics disabled: {e}")
            self.supabase = None

    async def log_context_metrics(self, metrics: ContextMetrics) -> bool:
        """Log comprehensive context engineering metrics"""
        if not self.supabase:
            logging.warning("Analytics disabled - Supabase not configured")
            return False

        try:
            metrics_data = {
                "org_id": metrics.org_id,
                "message_id": metrics.message_id,
                "query_original": metrics.query_original,
                "query_enhanced": metrics.query_enhanced,
                "documents_retrieved": metrics.documents_retrieved,
                "context_used": "",  # New field
                "retrieval_stats": {
                    "time_ms": metrics.retrieval_time_ms,
                    "sources_count": metrics.sources_count
                },
                "context_quality": {
                    "score": metrics.context_quality_score,
                    "length": metrics.context_length
                },
                "model_used": metrics.model_used
            }

            response = self.supabase.table(
                "context_analytics").insert(metrics_data).execute()
            return bool(response.data)

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logging.error("Failed to log context metrics: %s", e)
            return False

    async def get_performance_dashboard(self, org_id: str, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        if not self.supabase:
            logging.warning("Analytics disabled - Supabase not configured")
            return self._empty_dashboard()

        try:
            since_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            # Get analytics data
            response = self.supabase.table("context_analytics").select("*").eq(
                "org_id", org_id
            ).gte("created_at", since_date).execute()

            analytics_data = response.data or []

            if not analytics_data:
                return self._empty_dashboard()

            # Calculate key metrics
            dashboard = await self._calculate_dashboard_metrics(analytics_data, days)

            return dashboard

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logging.error("Failed to get performance dashboard: %s", e)
            return self._empty_dashboard()

    async def _calculate_dashboard_metrics(self, data: List[Dict], days: int) -> Dict[str, Any]:
        """Calculate comprehensive dashboard metrics"""

        # Basic statistics
        total_queries = len(data)
        response_times = [d.get("response_time_ms", 0)
                          for d in data if d.get("response_time_ms")]
        context_scores = [d.get("context_quality_score", 0)
                          for d in data if d.get("context_quality_score")]
        satisfaction_scores = [d.get("user_satisfaction", 0)
                               for d in data if d.get("user_satisfaction")]

        # Performance metrics
        avg_response_time = statistics.mean(
            response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[
            18] if len(response_times) > 5 else 0
        avg_context_quality = statistics.mean(
            context_scores) if context_scores else 0
        avg_satisfaction = statistics.mean(
            satisfaction_scores) if satisfaction_scores else 0

        # Usage patterns
        hourly_distribution = defaultdict(int)
        model_usage = Counter()
        source_diversity = []

        for record in data:
            # Hour distribution
            hour = datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")).hour
            hourly_distribution[hour] += 1

            # Model usage
            model_usage[record.get("model_used", "unknown")] += 1

            # Source diversity
            sources_count = record.get("sources_count", 0)
            source_diversity.append(sources_count)

        # Trends (compare with previous period)
        trends = await self._calculate_trends(data, days)

        # Quality analysis
        quality_analysis = self._analyze_quality_patterns(data)

        # Performance insights
        insights = self._generate_performance_insights(data)

        return {
            "summary": {
                "total_queries": total_queries,
                "avg_response_time_ms": round(avg_response_time, 2),
                "p95_response_time_ms": round(p95_response_time, 2),
                "avg_context_quality": round(avg_context_quality, 3),
                "avg_satisfaction": round(avg_satisfaction, 3),
                "period_days": days
            },
            "performance": {
                "response_time": {
                    "average": round(avg_response_time, 2),
                    "median": round(statistics.median(response_times), 2) if response_times else 0,
                    "p95": round(p95_response_time, 2),
                    "distribution": self._create_distribution(response_times, [1000, 2000, 3000, 5000])
                },
                "context_quality": {
                    "average": round(avg_context_quality, 3),
                    "distribution": self._create_distribution(context_scores, [0.5, 0.7, 0.8, 0.9])
                },
                "satisfaction": {
                    "average": round(avg_satisfaction, 3),
                    "total_feedback": len(satisfaction_scores),
                    "distribution": self._create_distribution(satisfaction_scores, [0.3, 0.5, 0.7, 0.9])
                }
            },
            "usage_patterns": {
                "hourly_distribution": dict(hourly_distribution),
                "model_usage": dict(model_usage),
                "avg_sources_per_query": round(statistics.mean(source_diversity), 2) if source_diversity else 0,
                "queries_per_day": round(total_queries / max(days, 1), 2)
            },
            "trends": trends,
            "quality_analysis": quality_analysis,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }

    async def _calculate_trends(self, current_data: List[Dict], days: int) -> Dict[str, Any]:
        """Calculate trends compared to previous period"""
        if not self.supabase:
            return {"trends_available": False}

        try:
            # Get previous period data
            prev_start = (datetime.utcnow() -
                          timedelta(days=days*2)).isoformat()
            prev_end = (datetime.utcnow() - timedelta(days=days)).isoformat()

            org_id = current_data[0]["org_id"] if current_data else None
            if not org_id:
                return {}

            prev_response = self.supabase.table("context_analytics").select("*").eq(
                "org_id", org_id
            ).gte("created_at", prev_start).lt("created_at", prev_end).execute()

            prev_data = prev_response.data or []

            if not prev_data:
                return {"trends_available": False}

            # Calculate trends
            current_avg_response = statistics.mean(
                [d.get("response_time_ms", 0) for d in current_data])
            prev_avg_response = statistics.mean(
                [d.get("response_time_ms", 0) for d in prev_data])

            current_avg_quality = statistics.mean(
                [d.get("context_quality_score", 0) for d in current_data])
            prev_avg_quality = statistics.mean(
                [d.get("context_quality_score", 0) for d in prev_data])

            return {
                "trends_available": True,
                "response_time_change": self._calculate_percentage_change(
                    prev_avg_response,
                    current_avg_response
                ),
                "quality_change": self._calculate_percentage_change(
                    prev_avg_quality, current_avg_quality
                ),
                "volume_change": self._calculate_percentage_change(len(prev_data), len(current_data))
            }

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logging.error("Error calculating trends: %s", e)
            return {"trends_available": False}

    def _analyze_quality_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in context quality"""

        quality_by_model = defaultdict(list)
        quality_by_sources = defaultdict(list)
        quality_by_time = defaultdict(list)

        for record in data:
            quality = record.get("context_quality_score", 0)
            model = record.get("model_used", "unknown")
            sources = record.get("sources_count", 0)
            hour = datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")).hour

            quality_by_model[model].append(quality)
            quality_by_sources[sources].append(quality)
            quality_by_time[hour].append(quality)

        return {
            "best_performing_model": max(quality_by_model.items(),
                                         key=lambda x: statistics.mean(x[1]) if x[1] else 0)[0] if quality_by_model else None,
            "optimal_source_count": max(quality_by_sources.items(),
                                        key=lambda x: statistics.mean(x[1]) if x[1] else 0)[0] if quality_by_sources else None,
            "peak_quality_hours": sorted(quality_by_time.items(),
                                         key=lambda x: statistics.mean(x[1]) if x[1] else 0, reverse=True)[:3]
        }

    def _generate_performance_insights(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Generate actionable performance insights"""
        insights = []

        if not data:
            return insights

        # Response time insights
        response_times = [d.get("response_time_ms", 0)
                          for d in data if d.get("response_time_ms")]
        if response_times:
            avg_response = statistics.mean(response_times)
            if avg_response > 5000:
                insights.append({
                    "type": "warning",
                    "category": "performance",
                    "title": "High Response Times Detected",
                    "description": f"Average response time is {avg_response:.0f}ms, which may impact user experience.",
                    "recommendation": "Consider optimizing retrieval count or switching to a faster model tier."
                })
            elif avg_response < 2000:
                insights.append({
                    "type": "success",
                    "category": "performance",
                    "title": "Excellent Response Times",
                    "description": f"Average response time of {avg_response:.0f}ms provides great user experience.",
                    "recommendation": "Current configuration is optimal for performance."
                })

        # Quality insights
        quality_scores = [d.get("context_quality_score", 0)
                          for d in data if d.get("context_quality_score")]
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            if avg_quality < 0.6:
                insights.append({
                    "type": "warning",
                    "category": "quality",
                    "title": "Low Context Quality",
                    "description": f"Average context quality is {avg_quality:.2f}, which may affect answer accuracy.",
                    "recommendation": "Increase retrieval count and enable semantic re-ranking."
                })
            elif avg_quality > 0.8:
                insights.append({
                    "type": "success",
                    "category": "quality",
                    "title": "High Context Quality",
                    "description": f"Excellent context quality of {avg_quality:.2f} ensures accurate responses.",
                    "recommendation": "Current context engineering settings are working well."
                })

        # Usage pattern insights
        model_usage = Counter(d.get("model_used", "unknown") for d in data)
        if len(model_usage) > 1:
            most_used = model_usage.most_common(1)[0]
            insights.append({
                "type": "info",
                "category": "usage",
                "title": "Model Usage Pattern",
                "description": f"Most frequently used model: {most_used[0]} ({most_used[1]} queries)",
                "recommendation": "Consider standardizing on this model if performance is satisfactory."
            })

        return insights

    def _create_distribution(self, values: List[float], thresholds: List[float]) -> Dict[str, int]:
        """Create distribution buckets for metrics"""
        if not values:
            return {}

        distribution = {}
        thresholds = sorted(thresholds)

        # Create buckets
        for i, threshold in enumerate(thresholds):
            if i == 0:
                key = f"< {threshold}"
                count = sum(1 for v in values if v < threshold)
            else:
                key = f"{thresholds[i-1]} - {threshold}"
                count = sum(
                    1 for v in values if thresholds[i-1] <= v < threshold)
            distribution[key] = count

        # Add final bucket
        final_key = f"> {thresholds[-1]}"
        distribution[final_key] = sum(1 for v in values if v >= thresholds[-1])

        return distribution

    def _calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 100.0 if new_value > 0 else 0.0
        return round(((new_value - old_value) / old_value) * 100, 2)

    def _empty_dashboard(self) -> Dict[str, Any]:
        """Return empty dashboard structure"""
        return {
            "summary": {
                "total_queries": 0,
                "avg_response_time_ms": 0,
                "avg_context_quality": 0,
                "avg_satisfaction": 0,
                "period_days": 7
            },
            "performance": {},
            "usage_patterns": {},
            "trends": {"trends_available": False},
            "quality_analysis": {},
            "insights": [],
            "generated_at": datetime.utcnow().isoformat()
        }

    async def get_query_analysis(self, org_id: str, query: str, days: int = 30) -> Dict[str, Any]:
        """Analyze similar queries and their performance"""
        if not self.supabase:
            logging.warning("Analytics disabled - Supabase not configured")
            return {"similar_queries_found": 0, "analysis": None}

        try:
            since_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            # Get similar queries (simple keyword matching for now)
            keywords = query.lower().split()
            similar_queries = []

            response = self.supabase.table("context_analytics").select("*").eq(
                "org_id", org_id
            ).gte("created_at", since_date).execute()

            for record in response.data or []:
                original_query = record.get("query_original", "").lower()
                if any(keyword in original_query for keyword in keywords):
                    similar_queries.append(record)

            if not similar_queries:
                return {"similar_queries_found": 0, "analysis": None}

            # Analyze similar queries
            avg_quality = statistics.mean(
                [q.get("context_quality_score", 0) for q in similar_queries])
            avg_response_time = statistics.mean(
                [q.get("response_time_ms", 0) for q in similar_queries])
            avg_satisfaction = statistics.mean([q.get(
                "user_satisfaction", 0) for q in similar_queries if q.get("user_satisfaction")])

            return {
                "similar_queries_found": len(similar_queries),
                "analysis": {
                    "avg_context_quality": round(avg_quality, 3),
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "avg_satisfaction": round(avg_satisfaction, 3) if similar_queries else None,
                    "most_common_sources": Counter([
                        source for q in similar_queries
                        for source in q.get("documents_retrieved", [])
                    ]).most_common(3)
                }
            }

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logging.error("Error in query analysis: %s", e)
            return {"similar_queries_found": 0, "analysis": None}

    async def export_analytics_data(
        self,
        org_id: str,
        start_date: datetime,
        end_date: datetime,
        export_format: str = "json"
    ) -> Dict[str, Any]:
        """Export analytics data for external analysis"""
        if not self.supabase:
            logging.warning("Analytics disabled - Supabase not configured")
            return {"format": export_format, "data": [], "count": 0, "error": "Analytics disabled"}

        try:
            response = self.supabase.table("context_analytics").select("*").eq(
                "org_id", org_id
            ).gte(
                "created_at", start_date.isoformat()
            ).lte(
                "created_at", end_date.isoformat()
            ).execute()

            data = response.data or []

            if export_format == "csv":
                # Convert to CSV format (simplified)
                csv_data = []
                for record in data:
                    csv_record = {
                        "timestamp": record["created_at"],
                        "query": record["query_original"],
                        "response_time_ms": record.get("response_time_ms", 0),
                        "context_quality": record.get("context_quality_score", 0),
                        "sources_count": record.get("sources_count", 0),
                        "model_used": record.get("model_used", ""),
                        "user_satisfaction": record.get("user_satisfaction", "")
                    }
                    csv_data.append(csv_record)
                return {"format": "csv", "data": csv_data, "count": len(csv_data)}

            return {"format": "json", "data": data, "count": len(data)}

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logging.error("Error exporting analytics data: %s", e)
            return {"format": export_format, "data": [], "count": 0, "error": str(e)}


# Global instance
context_analytics = ContextAnalytics()
