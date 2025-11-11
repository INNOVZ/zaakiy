"""
Analytics Service
Handles event tracking, metrics collection, and logging operations
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.services.shared import get_cache_service

if TYPE_CHECKING:
    from ..analytics.context_config import ContextEngineeringConfig
    from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


class AnalyticsServiceError(Exception):
    """Exception for analytics service errors"""


class AnalyticsService:
    """Handles analytics, logging, and performance tracking"""

    def __init__(
        self,
        org_id: str,
        supabase_client: "SupabaseClient",
        context_config: Optional["ContextEngineeringConfig"],
    ):
        self.org_id = org_id
        self.supabase = supabase_client
        self.context_config = context_config

    async def log_analytics(
        self,
        conversation_id: str,
        message_id: str,
        query_original: str,
        response_data: Dict[str, Any],
        processing_time: int,
    ) -> None:
        """Log comprehensive analytics data"""
        try:
            # Extract context quality score from nested structure
            context_quality = response_data.get("context_quality", {})
            context_quality_score = (
                context_quality.get("coverage_score", 0.5)
                if isinstance(context_quality, dict)
                else 0.5
            )

            # Extract retrieval stats
            retrieval_stats = response_data.get("retrieval_stats", {})
            retrieval_time_ms = (
                retrieval_stats.get("retrieval_time_ms", 0)
                if isinstance(retrieval_stats, dict)
                else 0
            )

            # Count sources
            sources = response_data.get("sources", [])
            sources_count = len(sources) if isinstance(sources, list) else 0

            # Prepare analytics data matching the actual database schema
            # Store nested data in JSONB fields as the schema expects
            # Get intent data from response
            intent_data = response_data.get("intent")
            intent_config = response_data.get("retrieval_stats", {}).get(
                "intent_config"
            )

            # Get enhanced queries and serialize if list (schema expects text or jsonb)
            enhanced_queries = response_data.get("enhanced_queries", [])
            if isinstance(enhanced_queries, list):
                # Store as JSON string for text field, or convert to jsonb-compatible format
                query_enhanced_text = (
                    json.dumps(enhanced_queries) if enhanced_queries else None
                )
            else:
                query_enhanced_text = enhanced_queries

            # Get context_used - schema expects jsonb, so store as object or null
            context_used_data = response_data.get("context_used", "")
            if isinstance(context_used_data, str):
                # If it's a string, store as JSON object with text field, or as null if empty
                context_used_jsonb = (
                    {"text": context_used_data} if context_used_data else None
                )
            else:
                # If it's already a dict/object, use it directly
                context_used_jsonb = context_used_data if context_used_data else None

            # Build retrieval_stats with intent data
            retrieval_stats_data = {
                "sources_used": sources_count,
                "retrieval_time_ms": retrieval_time_ms,
                "candidates_found": sources_count,
                "context_length": len(response_data.get("context_used", "")),
            }

            # Store intent data in retrieval_stats JSONB field (since schema doesn't have dedicated intent field)
            if intent_data:
                retrieval_stats_data["intent"] = intent_data
            if intent_config:
                retrieval_stats_data["intent_config"] = intent_config

            # Also store intent retrieval config if available
            retrieval_stats_from_response = response_data.get("retrieval_stats", {})
            if "documents_retrieved" in retrieval_stats_from_response:
                retrieval_stats_data[
                    "documents_retrieved"
                ] = retrieval_stats_from_response["documents_retrieved"]

            context_data = {
                "message_id": message_id,
                "org_id": self.org_id,
                "query_original": query_original,
                "query_enhanced": query_enhanced_text,  # Store as JSON string for text field
                "documents_retrieved": sources,
                "context_used": context_used_jsonb,  # Store as JSONB object
                "retrieval_stats": retrieval_stats_data,  # Includes intent data
                "context_quality": {
                    "score": context_quality_score,
                    "coverage_score": context_quality_score,
                    **context_quality,  # Include any other quality metrics
                },
                "model_used": str(self.context_config.model_tier)
                if self.context_config
                else "default",
            }

            # If intent column exists in schema (after migration), also store intent there for easier querying
            # This allows backward compatibility - intent is stored in both places
            # Primary storage: retrieval_stats->intent (works with current schema)
            # Secondary storage: intent column (works after migration, enables better queries)
            if intent_data:
                # Try to store in dedicated intent column (if migration has been run)
                # If column doesn't exist, this will be ignored by Supabase
                try:
                    context_data["intent"] = intent_data
                except Exception as e:
                    # Column might not exist yet - that's okay, intent is already in retrieval_stats
                    logger.debug(f"Intent column might not exist in schema: {e}")

            # Execute sync supabase call
            result = (
                self.supabase.table("context_analytics").insert(context_data).execute()
            )

            if result.data:
                logger.info(f"Analytics logged successfully for message {message_id}")
            else:
                logger.warning(
                    f"Analytics insert returned no data for message {message_id}"
                )

        except Exception as e:
            logger.warning("Analytics logging failed: %s", e)

    def log_performance_metrics(
        self,
        operation: str,
        duration_ms: int,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log performance metrics for operations"""
        try:
            performance_data = {
                "org_id": self.org_id,
                "operation": operation,
                "duration_ms": duration_ms,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            # Store in performance_metrics table
            self.supabase.table("performance_metrics").insert(
                performance_data
            ).execute()

        except Exception as e:
            logger.warning("Performance metrics logging failed: %s", e)

    def track_user_feedback(
        self, message_id: str, rating: int, feedback_text: Optional[str] = None
    ) -> bool:
        """Track user feedback and update analytics"""
        try:
            # Get message to find conversation_id
            msg_response = (
                self.supabase.table("messages")
                .select("conversation_id")
                .eq("id", message_id)
                .execute()
            )

            if not msg_response.data:
                logger.warning("Message not found for feedback: %s", message_id)
                return False

            conversation_id = msg_response.data[0]["conversation_id"]

            feedback_data = {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "org_id": self.org_id,
                "rating": rating,
                "feedback_text": feedback_text,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            response = (
                self.supabase.table("conversation_feedback")
                .insert(feedback_data)
                .execute()
            )

            # Update analytics with user satisfaction
            if response.data:
                try:
                    satisfaction_score = 1.0 if rating > 0 else 0.0
                    update_result = (
                        self.supabase.table("context_analytics")
                        .update(
                            {
                                "user_satisfaction": satisfaction_score,
                                "feedback_text": feedback_text,
                            }
                        )
                        .eq("message_id", message_id)
                        .execute()
                    )

                    if update_result.data:
                        logger.info(
                            f"Updated user satisfaction for message {message_id}: {satisfaction_score}"
                        )
                    else:
                        logger.warning(
                            f"No analytics record found to update for message {message_id}"
                        )

                    logger.info(
                        "Feedback recorded for message %s: rating=%s",
                        message_id,
                        rating,
                    )

                except Exception as e:
                    logger.warning("Failed to update analytics with feedback: %s", e)

            return bool(response.data)

        except Exception as e:
            logger.error("Error tracking user feedback: %s", e)
            return False

    def get_performance_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get performance insights for this organization"""
        try:
            # Get performance metrics for the specified period
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get basic performance data
            performance_response = (
                self.supabase.table("performance_metrics")
                .select("*")
                .eq("org_id", self.org_id)
                .gte("timestamp", start_date)
                .execute()
            )

            performance_data = performance_response.data or []

            # Calculate basic insights
            total_operations = len(performance_data)
            successful_operations = len(
                [p for p in performance_data if p.get("success", True)]
            )
            success_rate = (
                (successful_operations / total_operations * 100)
                if total_operations > 0
                else 0
            )

            avg_duration = (
                sum(p.get("duration_ms", 0) for p in performance_data)
                / total_operations
                if total_operations > 0
                else 0
            )

            # Get feedback data
            feedback_response = (
                self.supabase.table("conversation_feedback")
                .select("rating")
                .eq("org_id", self.org_id)
                .gte("created_at", start_date)
                .execute()
            )

            feedback_data = feedback_response.data or []
            positive_feedback = len(
                [f for f in feedback_data if f.get("rating", 0) > 0]
            )
            total_feedback = len(feedback_data)
            satisfaction_rate = (
                (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
            )

            insights = {
                "period_days": days,
                "total_operations": total_operations,
                "success_rate_percent": round(success_rate, 2),
                "average_duration_ms": round(avg_duration, 2),
                "total_feedback_received": total_feedback,
                "satisfaction_rate_percent": round(satisfaction_rate, 2),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            return insights

        except Exception as e:
            logger.error("Error getting performance insights: %s", e)
            return {"error": str(e)}

    def track_retrieval_metrics(
        self,
        query: str,
        documents_retrieved: int,
        retrieval_time_ms: int,
        quality_score: float,
        strategy_used: str,
    ) -> None:
        """Track document retrieval metrics"""
        try:
            retrieval_data = {
                "org_id": self.org_id,
                "query": query,
                "documents_retrieved": documents_retrieved,
                "retrieval_time_ms": retrieval_time_ms,
                "quality_score": quality_score,
                "strategy_used": strategy_used,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.supabase.table("retrieval_metrics").insert(retrieval_data).execute()

        except Exception as e:
            logger.warning("Retrieval metrics tracking failed: %s", e)

    def track_context_quality(
        self,
        message_id: str,
        context_length: int,
        relevance_score: float,
        coverage_score: float,
        coherence_score: float,
    ) -> None:
        """Track context quality metrics"""
        try:
            quality_data = {
                "message_id": message_id,
                "org_id": self.org_id,
                "context_length": context_length,
                "relevance_score": relevance_score,
                "coverage_score": coverage_score,
                "coherence_score": coherence_score,
                "overall_quality": (relevance_score + coverage_score + coherence_score)
                / 3,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.supabase.table("context_quality_metrics").insert(
                quality_data
            ).execute()

        except Exception as e:
            logger.warning("Context quality tracking failed: %s", e)

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get a summary of analytics data for the organization"""
        try:
            # Get data for the last 30 days
            start_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            # Get conversation count
            conversations_response = (
                self.supabase.table("conversations")
                .select("id")
                .eq("org_id", self.org_id)
                .gte("created_at", start_date)
                .execute()
            )

            # Get message count
            messages_response = (
                self.supabase.table("context_analytics")
                .select("message_id")
                .eq("org_id", self.org_id)
                .execute()
            )

            # Get average quality scores
            quality_response = (
                self.supabase.table("context_quality_metrics")
                .select("overall_quality")
                .eq("org_id", self.org_id)
                .gte("timestamp", start_date)
                .execute()
            )

            quality_data = quality_response.data or []
            avg_quality = (
                sum(q.get("overall_quality", 0) for q in quality_data)
                / len(quality_data)
                if quality_data
                else 0
            )

            summary = {
                "conversations_last_30_days": len(conversations_response.data or []),
                "total_messages": len(messages_response.data or []),
                "average_quality_score": round(avg_quality, 3),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            return summary

        except Exception as e:
            logger.error("Error getting analytics summary: %s", e)
            return {"error": str(e)}

    async def get_intent_analytics(
        self, days: int = 7, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive intent analytics for the organization.

        Args:
            days: Number of days to look back
            use_cache: Whether to use cache for faster responses

        Returns:
            Dict containing:
            - intent_distribution: Distribution of intents
            - intent_performance: Performance metrics by intent
            - intent_confidence: Confidence distribution
            - detection_methods: Detection method breakdown
            - intent_trends: Trends over time
            - intent_fulfillment: Fulfillment rates by intent
        """
        try:
            # Check cache first (5 minute TTL for analytics)
            if use_cache:
                try:
                    cache_service_instance = get_cache_service()
                    if cache_service_instance and hasattr(
                        cache_service_instance, "get"
                    ):
                        cache_key = f"intent_analytics:{self.org_id}:{days}"
                        try:
                            # Cache service is async, so we await it
                            cached_result = await cache_service_instance.get(cache_key)
                            if cached_result:
                                logger.debug(
                                    f"Intent analytics cache HIT for org {self.org_id}, days {days}"
                                )
                                return cached_result
                        except Exception as cache_error:
                            logger.debug(
                                f"Cache get failed (non-critical): {cache_error}"
                            )
                except Exception as cache_init_error:
                    logger.debug(
                        f"Cache service not available (non-critical): {cache_init_error}"
                    )

            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get all analytics records with intent data
            # Try both intent column and retrieval_stats->intent for backward compatibility
            analytics_response = (
                self.supabase.table("context_analytics")
                .select("*")
                .eq("org_id", self.org_id)
                .gte("created_at", start_date)
                .execute()
            )

            analytics_data = analytics_response.data or []

            # Filter records with intent data (from either location)
            intent_records = []
            for record in analytics_data:
                # Check dedicated intent column first, then retrieval_stats->intent
                intent_data = record.get("intent") or (
                    record.get("retrieval_stats", {}).get("intent")
                    if isinstance(record.get("retrieval_stats"), dict)
                    else None
                )
                if intent_data:
                    intent_records.append(
                        {
                            **record,
                            "intent_data": intent_data,
                        }
                    )

            if not intent_records:
                return {
                    "summary": {
                        "total_queries": 0,
                        "queries_with_intent": 0,
                        "intent_coverage": 0.0,
                        "unique_intents": 0,
                        "period_days": days,
                    },
                    "intent_distribution": {},
                    "intent_performance": {},
                    "intent_confidence": {},
                    "detection_methods": {},
                    "intent_trends": [],
                    "intent_fulfillment": {},
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

            # Calculate intent distribution
            intent_distribution = {}
            intent_performance = {}
            intent_confidence = {}
            detection_methods = {}
            intent_trends = {}

            for record in intent_records:
                intent_data = record["intent_data"]
                primary_intent = intent_data.get("primary_intent", "unknown")
                confidence = float(intent_data.get("confidence", 0.0))
                detection_method = intent_data.get("detection_method", "unknown")

                # Intent distribution
                intent_distribution[primary_intent] = (
                    intent_distribution.get(primary_intent, 0) + 1
                )

                # Performance metrics
                if primary_intent not in intent_performance:
                    intent_performance[primary_intent] = {
                        "total_queries": 0,
                        "avg_retrieval_time_ms": [],
                        "avg_quality_score": [],
                        "avg_sources_used": [],
                        "avg_confidence": [],
                    }

                perf = intent_performance[primary_intent]
                perf["total_queries"] += 1
                perf["avg_confidence"].append(confidence)

                # Get retrieval stats
                retrieval_stats = record.get("retrieval_stats", {})
                if isinstance(retrieval_stats, dict):
                    if "retrieval_time_ms" in retrieval_stats:
                        perf["avg_retrieval_time_ms"].append(
                            float(retrieval_stats["retrieval_time_ms"])
                        )
                    if "sources_used" in retrieval_stats:
                        perf["avg_sources_used"].append(
                            float(retrieval_stats["sources_used"])
                        )

                # Get context quality
                context_quality = record.get("context_quality", {})
                if isinstance(context_quality, dict):
                    quality_score = context_quality.get("score") or context_quality.get(
                        "coverage_score"
                    )
                    if quality_score:
                        perf["avg_quality_score"].append(float(quality_score))

                # Confidence distribution
                confidence_level = self._get_confidence_level(confidence)
                intent_confidence[confidence_level] = (
                    intent_confidence.get(confidence_level, 0) + 1
                )

                # Detection methods
                detection_methods[detection_method] = (
                    detection_methods.get(detection_method, 0) + 1
                )

                # Trends (by date)
                record_date = record.get("created_at", "")[:10]  # Get date part
                if record_date:
                    if record_date not in intent_trends:
                        intent_trends[record_date] = {}
                    intent_trends[record_date][primary_intent] = (
                        intent_trends[record_date].get(primary_intent, 0) + 1
                    )

            # Calculate averages for performance metrics
            for intent, perf in intent_performance.items():
                perf["avg_retrieval_time_ms"] = (
                    sum(perf["avg_retrieval_time_ms"])
                    / len(perf["avg_retrieval_time_ms"])
                    if perf["avg_retrieval_time_ms"]
                    else 0
                )
                perf["avg_quality_score"] = (
                    sum(perf["avg_quality_score"]) / len(perf["avg_quality_score"])
                    if perf["avg_quality_score"]
                    else 0
                )
                perf["avg_sources_used"] = (
                    sum(perf["avg_sources_used"]) / len(perf["avg_sources_used"])
                    if perf["avg_sources_used"]
                    else 0
                )
                perf["avg_confidence"] = (
                    sum(perf["avg_confidence"]) / len(perf["avg_confidence"])
                    if perf["avg_confidence"]
                    else 0
                )

            # Get intent fulfillment rates (with user feedback)
            intent_fulfillment = self._get_intent_fulfillment(intent_records, days)

            # Convert trends to list format
            trends_list = [
                {
                    "date": date,
                    "intents": intents,
                    "total": sum(intents.values()),
                }
                for date, intents in sorted(intent_trends.items())
            ]

            total_queries = len(analytics_data)
            queries_with_intent = len(intent_records)
            intent_coverage = (
                (queries_with_intent / total_queries * 100) if total_queries > 0 else 0
            )

            result = {
                "summary": {
                    "total_queries": total_queries,
                    "queries_with_intent": queries_with_intent,
                    "intent_coverage": round(intent_coverage, 2),
                    "unique_intents": len(intent_distribution),
                    "period_days": days,
                },
                "intent_distribution": intent_distribution,
                "intent_performance": {
                    intent: {
                        "total_queries": perf["total_queries"],
                        "avg_retrieval_time_ms": round(
                            perf["avg_retrieval_time_ms"], 2
                        ),
                        "avg_quality_score": round(perf["avg_quality_score"], 3),
                        "avg_sources_used": round(perf["avg_sources_used"], 1),
                        "avg_confidence": round(perf["avg_confidence"], 3),
                    }
                    for intent, perf in intent_performance.items()
                },
                "intent_confidence": intent_confidence,
                "detection_methods": detection_methods,
                "intent_trends": trends_list,
                "intent_fulfillment": intent_fulfillment,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Cache result for 5 minutes (300 seconds)
            if use_cache:
                try:
                    cache_service_instance = get_cache_service()
                    if cache_service_instance and hasattr(
                        cache_service_instance, "set"
                    ):
                        cache_key = f"intent_analytics:{self.org_id}:{days}"
                        try:
                            # Cache service is async, so we await it
                            await cache_service_instance.set(
                                cache_key, result, ttl_seconds=300
                            )
                            logger.debug(
                                f"Cached intent analytics for org {self.org_id}, days {days}"
                            )
                        except Exception as cache_error:
                            logger.debug(
                                f"Cache set failed (non-critical): {cache_error}"
                            )
                except Exception as cache_init_error:
                    logger.debug(
                        f"Cache service not available (non-critical): {cache_init_error}"
                    )

            return result

        except Exception as e:
            logger.error("Error getting intent analytics: %s", e, exc_info=True)
            return {"error": str(e)}

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "unknown"

    def _get_intent_fulfillment(
        self, intent_records: List[Dict[str, Any]], days: int
    ) -> Dict[str, Any]:
        """Get intent fulfillment rates based on user feedback"""
        try:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get feedback data
            feedback_response = (
                self.supabase.table("conversation_feedback")
                .select("message_id, rating, feedback_text")
                .eq("org_id", self.org_id)
                .gte("created_at", start_date)
                .execute()
            )

            feedback_data = {f["message_id"]: f for f in (feedback_response.data or [])}

            # Calculate fulfillment by intent
            fulfillment_by_intent = {}

            for record in intent_records:
                message_id = record.get("message_id")
                if not message_id:
                    continue

                intent_data = record["intent_data"]
                primary_intent = intent_data.get("primary_intent", "unknown")

                if primary_intent not in fulfillment_by_intent:
                    fulfillment_by_intent[primary_intent] = {
                        "total_queries": 0,
                        "queries_with_feedback": 0,
                        "positive_feedback": 0,
                        "avg_rating": [],
                        "satisfaction_rate": 0.0,
                    }

                fulfillment = fulfillment_by_intent[primary_intent]
                fulfillment["total_queries"] += 1

                if message_id in feedback_data:
                    feedback = feedback_data[message_id]
                    fulfillment["queries_with_feedback"] += 1
                    rating = feedback.get("rating", 0)
                    if rating > 0:
                        fulfillment["avg_rating"].append(rating)
                    if rating >= 4:  # Positive feedback
                        fulfillment["positive_feedback"] += 1

            # Calculate satisfaction rates
            for intent, fulfillment in fulfillment_by_intent.items():
                if fulfillment["queries_with_feedback"] > 0:
                    fulfillment["satisfaction_rate"] = round(
                        (
                            fulfillment["positive_feedback"]
                            / fulfillment["queries_with_feedback"]
                        )
                        * 100,
                        2,
                    )
                    fulfillment["avg_rating"] = (
                        round(
                            sum(fulfillment["avg_rating"])
                            / len(fulfillment["avg_rating"]),
                            2,
                        )
                        if fulfillment["avg_rating"]
                        else 0.0
                    )
                else:
                    fulfillment["avg_rating"] = 0.0

            return fulfillment_by_intent

        except Exception as e:
            logger.warning("Error getting intent fulfillment: %s", e)
            return {}

    async def get_intent_details(
        self, intent_type: str, days: int = 7, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed analytics for a specific intent type.

        Args:
            intent_type: The intent type to analyze
            days: Number of days to look back
            use_cache: Whether to use cache for faster responses

        Returns:
            Dict containing detailed intent analytics
        """
        try:
            # Check cache first (5 minute TTL for analytics)
            if use_cache:
                try:
                    cache_service_instance = get_cache_service()
                    if cache_service_instance and hasattr(
                        cache_service_instance, "get"
                    ):
                        cache_key = f"intent_details:{self.org_id}:{intent_type}:{days}"
                        try:
                            # Cache service is async, so we await it
                            cached_result = await cache_service_instance.get(cache_key)
                            if cached_result:
                                logger.debug(
                                    f"Intent details cache HIT for org {self.org_id}, intent {intent_type}, days {days}"
                                )
                                return cached_result
                        except Exception as cache_error:
                            logger.debug(
                                f"Cache get failed (non-critical): {cache_error}"
                            )
                except Exception as cache_init_error:
                    logger.debug(
                        f"Cache service not available (non-critical): {cache_init_error}"
                    )

            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get analytics records for this intent
            analytics_response = (
                self.supabase.table("context_analytics")
                .select("*")
                .eq("org_id", self.org_id)
                .gte("created_at", start_date)
                .execute()
            )

            analytics_data = analytics_response.data or []

            # Filter records with matching intent
            intent_records = []
            for record in analytics_data:
                intent_data = record.get("intent") or (
                    record.get("retrieval_stats", {}).get("intent")
                    if isinstance(record.get("retrieval_stats"), dict)
                    else None
                )
                if intent_data and intent_data.get("primary_intent") == intent_type:
                    intent_records.append(record)

            if not intent_records:
                return {
                    "intent_type": intent_type,
                    "total_queries": 0,
                    "error": "No data found for this intent type",
                }

            # Calculate detailed metrics
            confidences = []
            retrieval_times = []
            quality_scores = []
            sources_used = []
            detection_methods = {}

            for record in intent_records:
                intent_data = record.get("intent") or record.get(
                    "retrieval_stats", {}
                ).get("intent")
                if intent_data:
                    confidences.append(float(intent_data.get("confidence", 0.0)))
                    detection_method = intent_data.get("detection_method", "unknown")
                    detection_methods[detection_method] = (
                        detection_methods.get(detection_method, 0) + 1
                    )

                retrieval_stats = record.get("retrieval_stats", {})
                if isinstance(retrieval_stats, dict):
                    if "retrieval_time_ms" in retrieval_stats:
                        retrieval_times.append(
                            float(retrieval_stats["retrieval_time_ms"])
                        )
                    if "sources_used" in retrieval_stats:
                        sources_used.append(float(retrieval_stats["sources_used"]))

                context_quality = record.get("context_quality", {})
                if isinstance(context_quality, dict):
                    quality_score = context_quality.get("score") or context_quality.get(
                        "coverage_score"
                    )
                    if quality_score:
                        quality_scores.append(float(quality_score))

            result = {
                "intent_type": intent_type,
                "total_queries": len(intent_records),
                "confidence": {
                    "average": round(sum(confidences) / len(confidences), 3)
                    if confidences
                    else 0,
                    "min": round(min(confidences), 3) if confidences else 0,
                    "max": round(max(confidences), 3) if confidences else 0,
                    "distribution": self._get_confidence_distribution(confidences),
                },
                "performance": {
                    "avg_retrieval_time_ms": round(
                        sum(retrieval_times) / len(retrieval_times), 2
                    )
                    if retrieval_times
                    else 0,
                    "avg_quality_score": round(
                        sum(quality_scores) / len(quality_scores), 3
                    )
                    if quality_scores
                    else 0,
                    "avg_sources_used": round(sum(sources_used) / len(sources_used), 1)
                    if sources_used
                    else 0,
                },
                "detection_methods": detection_methods,
                "period_days": days,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Cache result for 5 minutes (300 seconds)
            if use_cache:
                try:
                    cache_service_instance = get_cache_service()
                    if cache_service_instance and hasattr(
                        cache_service_instance, "set"
                    ):
                        cache_key = f"intent_details:{self.org_id}:{intent_type}:{days}"
                        try:
                            # Cache service is async, so we await it
                            await cache_service_instance.set(
                                cache_key, result, ttl_seconds=300
                            )
                            logger.debug(
                                f"Cached intent details for org {self.org_id}, intent {intent_type}, days {days}"
                            )
                        except Exception as cache_error:
                            logger.debug(
                                f"Cache set failed (non-critical): {cache_error}"
                            )
                except Exception as cache_init_error:
                    logger.debug(
                        f"Cache service not available (non-critical): {cache_init_error}"
                    )

            return result

        except Exception as e:
            logger.error("Error getting intent details: %s", e)
            return {"error": str(e)}

    def _get_confidence_distribution(self, confidences: List[float]) -> Dict[str, int]:
        """Get confidence distribution"""
        distribution = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for confidence in confidences:
            level = self._get_confidence_level(confidence)
            distribution[level] = distribution.get(level, 0) + 1
        return distribution
