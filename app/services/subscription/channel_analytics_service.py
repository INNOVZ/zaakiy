"""Channel analytics service for multi-channel subscription tracking."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from supabase import Client

from app.models.subscription import (Channel, ChannelUsageStats, SubscriptionAnalytics,
                                     SubscriptionPlan)

logger = logging.getLogger(__name__)


class ChannelAnalyticsService:
    """Service for managing channel-specific analytics and insights."""

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client

    async def get_subscription_analytics(
        self, subscription_id: str, days_back: int = 30
    ) -> SubscriptionAnalytics:
        """
        Get comprehensive analytics for a subscription including channel breakdown.

        Args:
            subscription_id: Subscription ID
            days_back: Number of days to analyze

        Returns:
            SubscriptionAnalytics with detailed channel breakdown
        """
        try:
            # Get subscription details
            subscription_result = (
                self.supabase.table("subscriptions")
                .select("*")
                .eq("id", subscription_id)
                .execute()
            )

            if not subscription_result.data:
                raise Exception("Subscription not found")

            subscription = subscription_result.data[0]

            # Get channel usage stats
            channel_usage = await self._get_channel_usage_stats(
                subscription_id, days_back
            )

            # Get daily usage for the period
            daily_usage = await self._get_daily_usage(subscription_id, days_back)

            # Get hourly distribution
            hourly_distribution = await self._get_hourly_distribution(
                subscription_id, days_back
            )

            # Calculate metrics
            total_tokens_used = sum(stats.tokens_used for stats in channel_usage)
            usage_percentage = (
                (total_tokens_used / subscription["monthly_token_limit"]) * 100
                if subscription["monthly_token_limit"] > 0
                else 0
            )

            # Find most/least active channels
            most_active = (
                max(channel_usage, key=lambda x: x.tokens_used)
                if channel_usage
                else None
            )
            least_active = (
                min(channel_usage, key=lambda x: x.tokens_used)
                if channel_usage
                else None
            )

            # Calculate growth rate (compare with previous period)
            growth_rate = await self._calculate_growth_rate(subscription_id, days_back)

            # Calculate days remaining in billing cycle
            billing_end = datetime.fromisoformat(
                subscription["billing_cycle_end"].replace("Z", "+00:00")
            )
            days_remaining = max(
                0,
                (
                    billing_end - datetime.utcnow().replace(tzinfo=billing_end.tzinfo)
                ).days,
            )

            return SubscriptionAnalytics(
                subscription_id=subscription_id,
                entity_id=subscription["entity_id"],
                entity_type=subscription["entity_type"],
                plan=SubscriptionPlan(subscription["plan"]),
                total_tokens_used=total_tokens_used,
                total_tokens_limit=subscription["monthly_token_limit"],
                usage_percentage=usage_percentage,
                channel_usage=channel_usage,
                daily_usage=daily_usage,
                hourly_distribution=hourly_distribution,
                most_active_channel=most_active.channel
                if most_active
                else Channel.WEBSITE,
                least_active_channel=least_active.channel
                if least_active
                else Channel.WEBSITE,
                growth_rate=growth_rate,
                billing_cycle_start=datetime.fromisoformat(
                    subscription["billing_cycle_start"]
                ),
                billing_cycle_end=billing_end,
                days_remaining=days_remaining,
            )

        except Exception as e:
            logger.error("Failed to get subscription analytics: %s", str(e))
            raise Exception(f"Failed to get subscription analytics: {str(e)}")

    async def _get_channel_usage_stats(
        self, subscription_id: str, days_back: int
    ) -> List[ChannelUsageStats]:
        """Get usage statistics by channel."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)

            # Query the channel_usage_analytics table directly
            result = (
                self.supabase.table("channel_usage_analytics")
                .select("*")
                .eq("subscription_id", subscription_id)
                .gte("date", start_date.isoformat())
                .lte("date", end_date.isoformat())
                .execute()
            )

            if not result.data:
                return []

            # Group by channel and aggregate data
            channel_data = {}
            for row in result.data:
                channel = row["channel"]
                if channel not in channel_data:
                    channel_data[channel] = {
                        "tokens_used": 0,
                        "message_count": 0,
                        "unique_users": set(),
                        "total_tokens": 0,
                        "total_messages": 0,
                    }

                channel_data[channel]["tokens_used"] += row["tokens_used"]
                channel_data[channel]["message_count"] += row["message_count"]
                channel_data[channel]["unique_users"].add(row.get("unique_users", 0))
                channel_data[channel]["total_tokens"] += row["tokens_used"]
                channel_data[channel]["total_messages"] += row["message_count"]

            channel_stats = []
            for channel, data in channel_data.items():
                # Calculate average tokens per message
                avg_tokens = 0
                if data["total_messages"] > 0:
                    avg_tokens = data["total_tokens"] / data["total_messages"]

                # Get peak usage hour for this channel
                peak_hour = await self._get_peak_usage_hour(
                    subscription_id, channel, days_back
                )

                channel_stats.append(
                    ChannelUsageStats(
                        channel=Channel(channel),
                        tokens_used=data["tokens_used"],
                        message_count=data["message_count"],
                        unique_users=len(data["unique_users"]),
                        avg_tokens_per_message=avg_tokens,
                        peak_usage_hour=peak_hour,
                    )
                )

            return channel_stats

        except Exception as e:
            logger.error("Failed to get channel usage stats: %s", str(e))
            return []

    async def _get_daily_usage(
        self, subscription_id: str, days_back: int
    ) -> Dict[str, int]:
        """Get daily token usage for the specified period."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)

            result = (
                self.supabase.table("channel_usage_analytics")
                .select("date, tokens_used")
                .eq("subscription_id", subscription_id)
                .gte("date", start_date.date().isoformat())
                .execute()
            )

            daily_usage = {}
            for row in result.data:
                date_str = row["date"]
                if date_str in daily_usage:
                    daily_usage[date_str] += row["tokens_used"]
                else:
                    daily_usage[date_str] = row["tokens_used"]

            return daily_usage

        except Exception as e:
            logger.error("Failed to get daily usage: %s", str(e))
            return {}

    async def _get_hourly_distribution(
        self, subscription_id: str, days_back: int
    ) -> Dict[int, int]:
        """Get hourly distribution of token usage."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)

            result = (
                self.supabase.table("token_usage_logs")
                .select("timestamp, tokens_consumed")
                .eq("subscription_id", subscription_id)
                .gte("timestamp", start_date.isoformat())
                .execute()
            )

            hourly_distribution = {hour: 0 for hour in range(24)}

            for row in result.data:
                timestamp = datetime.fromisoformat(row["timestamp"])
                hour = timestamp.hour
                hourly_distribution[hour] += row["tokens_consumed"]

            return hourly_distribution

        except Exception as e:
            logger.error("Failed to get hourly distribution: %s", str(e))
            return {hour: 0 for hour in range(24)}

    async def _get_peak_usage_hour(
        self, subscription_id: str, channel: str, days_back: int
    ) -> Optional[int]:
        """Get the peak usage hour for a specific channel."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)

            result = (
                self.supabase.table("token_usage_logs")
                .select("timestamp, tokens_consumed")
                .eq("subscription_id", subscription_id)
                .eq("channel", channel)
                .gte("timestamp", start_date.isoformat())
                .execute()
            )

            hourly_usage = {hour: 0 for hour in range(24)}

            for row in result.data:
                timestamp = datetime.fromisoformat(row["timestamp"])
                hour = timestamp.hour
                hourly_usage[hour] += row["tokens_consumed"]

            if not any(hourly_usage.values()):
                return None

            return max(hourly_usage, key=hourly_usage.get)

        except Exception as e:
            logger.error("Failed to get peak usage hour: %s", str(e))
            return None

    async def _calculate_growth_rate(
        self, subscription_id: str, days_back: int
    ) -> float:
        """Calculate growth rate compared to previous period."""
        try:
            # Current period
            current_start = datetime.utcnow() - timedelta(days=days_back)
            current_result = (
                self.supabase.table("channel_usage_analytics")
                .select("tokens_used")
                .eq("subscription_id", subscription_id)
                .gte("date", current_start.date().isoformat())
                .execute()
            )

            current_usage = sum(row["tokens_used"] for row in current_result.data)

            # Previous period
            previous_start = current_start - timedelta(days=days_back)
            previous_end = current_start

            previous_result = (
                self.supabase.table("channel_usage_analytics")
                .select("tokens_used")
                .eq("subscription_id", subscription_id)
                .gte("date", previous_start.date().isoformat())
                .lt("date", previous_end.date().isoformat())
                .execute()
            )

            previous_usage = sum(row["tokens_used"] for row in previous_result.data)

            if previous_usage == 0:
                return 100.0 if current_usage > 0 else 0.0

            growth_rate = ((current_usage - previous_usage) / previous_usage) * 100
            return round(growth_rate, 2)

        except Exception as e:
            logger.error("Failed to calculate growth rate: %s", str(e))
            return 0.0

    async def get_channel_performance_comparison(
        self, subscription_id: str, days_back: int = 30
    ) -> Dict[str, Dict]:
        """
        Compare performance across channels.

        Returns:
            Dictionary with channel performance metrics
        """
        try:
            channel_stats = await self._get_channel_usage_stats(
                subscription_id, days_back
            )

            if not channel_stats:
                return {}

            total_tokens = sum(stats.tokens_used for stats in channel_stats)
            total_messages = sum(stats.message_count for stats in channel_stats)

            comparison = {}

            for stats in channel_stats:
                efficiency_score = (
                    stats.avg_tokens_per_message
                    / (total_tokens / total_messages if total_messages > 0 else 1)
                ) * 100

                comparison[stats.channel.value] = {
                    "tokens_used": stats.tokens_used,
                    "message_count": stats.message_count,
                    "unique_users": stats.unique_users,
                    "avg_tokens_per_message": stats.avg_tokens_per_message,
                    "usage_share_percentage": (stats.tokens_used / total_tokens * 100)
                    if total_tokens > 0
                    else 0,
                    "efficiency_score": round(efficiency_score, 2),
                    "peak_usage_hour": stats.peak_usage_hour,
                    "performance_rating": self._calculate_performance_rating(
                        stats, total_tokens
                    ),
                }

            return comparison

        except Exception as e:
            logger.error("Failed to get channel performance comparison: %s", str(e))
            return {}

    def _calculate_performance_rating(
        self, stats: ChannelUsageStats, total_tokens: int
    ) -> str:
        """Calculate a performance rating for a channel."""
        usage_share = (
            (stats.tokens_used / total_tokens * 100) if total_tokens > 0 else 0
        )

        if usage_share >= 40:
            return "Excellent"
        elif usage_share >= 25:
            return "Good"
        elif usage_share >= 10:
            return "Average"
        elif usage_share >= 5:
            return "Below Average"
        else:
            return "Poor"

    async def get_channel_trends(
        self, subscription_id: str, channel: Channel, days_back: int = 30
    ) -> Dict[str, List]:
        """
        Get usage trends for a specific channel.

        Returns:
            Dictionary with daily trends and predictions
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)

            result = (
                self.supabase.table("channel_usage_analytics")
                .select("date, tokens_used, message_count, unique_users")
                .eq("subscription_id", subscription_id)
                .eq("channel", channel.value)
                .gte("date", start_date.date().isoformat())
                .order("date")
                .execute()
            )

            if not result.data:
                return {"daily_tokens": [], "daily_messages": [], "daily_users": []}

            daily_tokens = []
            daily_messages = []
            daily_users = []

            for row in result.data:
                daily_tokens.append({"date": row["date"], "value": row["tokens_used"]})
                daily_messages.append(
                    {"date": row["date"], "value": row["message_count"]}
                )
                daily_users.append({"date": row["date"], "value": row["unique_users"]})

            return {
                "daily_tokens": daily_tokens,
                "daily_messages": daily_messages,
                "daily_users": daily_users,
                "trend_direction": self._calculate_trend_direction(daily_tokens),
                "average_daily_tokens": sum(item["value"] for item in daily_tokens)
                / len(daily_tokens)
                if daily_tokens
                else 0,
            }

        except Exception as e:
            logger.error("Failed to get channel trends: %s", str(e))
            return {"daily_tokens": [], "daily_messages": [], "daily_users": []}

    def _calculate_trend_direction(self, daily_data: List[Dict]) -> str:
        """Calculate if the trend is increasing, decreasing, or stable."""
        if len(daily_data) < 2:
            return "stable"

        recent_avg = sum(item["value"] for item in daily_data[-7:]) / min(
            7, len(daily_data)
        )
        older_avg = sum(item["value"] for item in daily_data[:-7]) / max(
            1, len(daily_data) - 7
        )

        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
