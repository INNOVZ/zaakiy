"""
Analytics Service
Handles event tracking, metrics collection, and logging operations
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

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
            context_quality_score = context_quality.get("coverage_score", 0.5) if isinstance(context_quality, dict) else 0.5
            
            # Extract retrieval stats
            retrieval_stats = response_data.get("retrieval_stats", {})
            retrieval_time_ms = retrieval_stats.get("retrieval_time_ms", 0) if isinstance(retrieval_stats, dict) else 0
            
            # Count sources
            sources = response_data.get("sources", [])
            sources_count = len(sources) if isinstance(sources, list) else 0
            
            # Prepare analytics data matching the actual database schema
            # Store nested data in JSONB fields as the schema expects
            context_data = {
                "message_id": message_id,
                "org_id": self.org_id,
                "query_original": query_original,
                "query_enhanced": response_data.get("enhanced_queries", []),
                "documents_retrieved": sources,
                "context_used": response_data.get("context_used", ""),
                "retrieval_stats": {
                    "sources_used": sources_count,  # Match existing field name
                    "retrieval_time_ms": retrieval_time_ms,  # Match existing field name
                    "candidates_found": sources_count,  # Add candidates found
                    "context_length": len(response_data.get("context_used", "")),  # Add context length
                },
                "context_quality": {
                    "score": context_quality_score,  # Store quality score in context_quality
                    "coverage_score": context_quality_score,  # For backward compatibility
                    **context_quality  # Include any other quality metrics
                },
                "model_used": str(self.context_config.model_tier) if self.context_config else "default",
            }

            # Execute sync supabase call
            result = self.supabase.table("context_analytics").insert(context_data).execute()
            
            if result.data:
                logger.info(f"Analytics logged successfully for message {message_id}")
            else:
                logger.warning(f"Analytics insert returned no data for message {message_id}")

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
                    update_result = self.supabase.table("context_analytics").update(
                        {
                            "user_satisfaction": satisfaction_score,
                            "feedback_text": feedback_text,
                        }
                    ).eq("message_id", message_id).execute()
                    
                    if update_result.data:
                        logger.info(f"Updated user satisfaction for message {message_id}: {satisfaction_score}")
                    else:
                        logger.warning(f"No analytics record found to update for message {message_id}")

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
