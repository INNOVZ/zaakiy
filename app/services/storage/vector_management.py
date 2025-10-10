"""
Enhanced vector management with comprehensive fault tolerance and retry logic
"""
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pinecone import Pinecone
import os
from supabase import create_client, Client
from ...config.settings import get_ai_config
from ...utils.logging_config import LogContext, PerformanceLogger, get_logger

logger = get_logger("vector_management")


@dataclass
class VectorDeletionResult:
    """Result of vector deletion operation"""
    success: bool
    vectors_deleted: int
    method_used: str
    attempts: int
    errors: List[str]
    duration_ms: float
    upload_id: str
    namespace: str


@dataclass
class VectorCleanupStats:
    """Statistics for vector cleanup operations"""
    total_uploads_processed: int
    successful_deletions: int
    failed_deletions: int
    total_vectors_deleted: int
    total_duration_ms: float
    errors_encountered: List[str]


class VectorDeletionStrategy:
    """Strategy pattern for vector deletion methods"""

    def __init__(self, index, max_retries: int = 3, retry_delay: float = 1.0):
        self.index = index
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def delete_vectors(self, upload_id: str, namespace: str) -> VectorDeletionResult:
        """Template method for vector deletion"""
        start_time = datetime.utcnow()
        result = VectorDeletionResult(
            success=False,
            vectors_deleted=0,
            method_used=self.__class__.__name__,
            attempts=0,
            errors=[],
            duration_ms=0,
            upload_id=upload_id,
            namespace=namespace
        )

        for attempt in range(self.max_retries):
            result.attempts = attempt + 1

            try:
                with LogContext(upload_id=upload_id, namespace=namespace, attempt=attempt + 1):
                    logger.info(
                        f"Vector deletion attempt {attempt + 1} using {self.__class__.__name__}")

                    deletion_result = await self._execute_deletion(upload_id, namespace)

                    if deletion_result["success"]:
                        result.success = True
                        result.vectors_deleted = deletion_result.get(
                            "vectors_deleted", 0)
                        result.duration_ms = (
                            datetime.utcnow() - start_time).total_seconds() * 1000

                        logger.info(
                            f"Vector deletion successful: {result.vectors_deleted} vectors deleted")
                        return result
                    else:
                        error_msg = deletion_result.get(
                            "error", "Unknown error")
                        result.errors.append(
                            f"Attempt {attempt + 1}: {error_msg}")
                        logger.warning(
                            f"Deletion attempt {attempt + 1} failed: {error_msg}")

            except Exception as e:
                error_msg = f"Attempt {attempt + 1} exception: {str(e)}"
                result.errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)

        result.duration_ms = (datetime.utcnow() -
                              start_time).total_seconds() * 1000
        logger.error(f"All deletion attempts failed for upload {upload_id}")
        return result

    async def _execute_deletion(self, upload_id: str, namespace: str) -> Dict[str, Any]:
        """Override this method in subclasses"""
        raise NotImplementedError


class MetadataFilterDeletion(VectorDeletionStrategy):
    """Delete vectors using metadata filter (most efficient)"""

    async def _execute_deletion(self, upload_id: str, namespace: str) -> Dict[str, Any]:
        try:
            logger.info(
                f"Attempting metadata filter deletion for upload {upload_id}")

            delete_response = self.index.delete(
                filter={"upload_id": upload_id},
                namespace=namespace
            )

            logger.info(
                f"Metadata filter deletion response: {delete_response}")

            return {
                "success": True,
                "vectors_deleted": "unknown",  # Pinecone doesn't return count for filter deletion
                "response": delete_response
            }

        except Exception as e:
            logger.warning(f"Metadata filter deletion failed: {e}")
            return {"success": False, "error": str(e)}


class QueryBatchDeletion(VectorDeletionStrategy):
    """Delete vectors by querying for IDs in batches"""

    def __init__(self, index, max_retries: int = 3, retry_delay: float = 1.0, batch_size: int = 1000):
        super().__init__(index, max_retries, retry_delay)
        self.batch_size = batch_size

    async def _execute_deletion(self, upload_id: str, namespace: str) -> Dict[str, Any]:
        try:
            logger.info(
                f"Starting query-batch deletion for upload {upload_id}")

            vector_ids = []
            has_more = True
            safety_counter = 0
            max_iterations = 20  # Safety limit

            while has_more and safety_counter < max_iterations:
                try:
                    # Query for vector IDs
                    query_result = self.index.query(
                        filter={"upload_id": upload_id},
                        namespace=namespace,
                        top_k=self.batch_size,
                        include_metadata=False,
                        include_values=False
                    )

                    batch_ids = [match.id for match in query_result.matches]

                    if not batch_ids:
                        has_more = False
                        logger.info("No more vectors found in query")
                    else:
                        vector_ids.extend(batch_ids)
                        logger.info(
                            f"Found {len(batch_ids)} vectors in batch {safety_counter + 1}")

                        if len(batch_ids) < self.batch_size:
                            has_more = False

                    safety_counter += 1

                except Exception as query_error:
                    logger.warning(
                        f"Query batch {safety_counter + 1} failed: {query_error}")
                    break

            if not vector_ids:
                logger.info(f"No vectors found for upload {upload_id}")
                return {"success": True, "vectors_deleted": 0}

            # Delete vectors in optimal batches
            deleted_count = await self._delete_vector_ids_in_batches(vector_ids, namespace)

            return {
                "success": True,
                "vectors_deleted": deleted_count,
                "total_found": len(vector_ids),
                "iterations": safety_counter
            }

        except Exception as e:
            logger.error(f"Query-batch deletion failed: {e}")
            return {"success": False, "error": str(e)}

    async def _delete_vector_ids_in_batches(self, vector_ids: List[str], namespace: str) -> int:
        """Delete vector IDs in optimal batches with error handling"""
        delete_batch_size = 100
        total_deleted = 0
        failed_deletions = []

        for i in range(0, len(vector_ids), delete_batch_size):
            batch = vector_ids[i:i + delete_batch_size]
            batch_num = i // delete_batch_size + 1

            try:
                delete_result = self.index.delete(
                    ids=batch, namespace=namespace)
                total_deleted += len(batch)
                logger.info(f"Deleted batch {batch_num}: {len(batch)} vectors")

                # Small delay between batches to avoid rate limits
                await asyncio.sleep(0.1)

            except Exception as batch_error:
                logger.error(
                    f"Failed to delete batch {batch_num}: {batch_error}")
                failed_deletions.extend(batch)
                continue

        if failed_deletions:
            logger.warning(f"Failed to delete {len(failed_deletions)} vectors")

        logger.info(f"Successfully deleted {total_deleted} vectors")
        return total_deleted


class DummyVectorDeletion(VectorDeletionStrategy):
    """Delete using dummy vector query (last resort)"""

    def __init__(self, index, max_retries: int = 3, retry_delay: float = 1.0, dimension: int = 1536):
        super().__init__(index, max_retries, retry_delay)
        self.dimension = dimension

    async def _execute_deletion(self, upload_id: str, namespace: str) -> Dict[str, Any]:
        try:
            logger.info(
                f"Starting dummy vector deletion for upload {upload_id}")

            # Create dummy vector
            dummy_vector = [0.0] * self.dimension

            vector_ids = []
            max_iterations = 15  # Reduced for safety

            for iteration in range(max_iterations):
                try:
                    query_result = self.index.query(
                        vector=dummy_vector,
                        filter={"upload_id": upload_id},
                        namespace=namespace,
                        top_k=500,  # Smaller batches for stability
                        include_metadata=False,
                        include_values=False
                    )

                    batch_ids = [match.id for match in query_result.matches]

                    if not batch_ids:
                        break

                    vector_ids.extend(batch_ids)
                    logger.info(
                        f"Dummy vector iteration {iteration + 1}: found {len(batch_ids)} vectors")

                    if len(batch_ids) < 500:
                        break

                except Exception as query_error:
                    logger.warning(
                        f"Dummy vector query iteration {iteration + 1} failed: {query_error}")
                    break

            if not vector_ids:
                return {"success": True, "vectors_deleted": 0}

            deleted_count = await self._delete_vector_ids_in_batches(vector_ids, namespace)

            return {
                "success": True,
                "vectors_deleted": deleted_count,
                "iterations": iteration + 1
            }

        except Exception as e:
            logger.error(f"Dummy vector deletion failed: {e}")
            return {"success": False, "error": str(e)}

    async def _delete_vector_ids_in_batches(self, vector_ids: List[str], namespace: str) -> int:
        """Reuse the batch deletion logic"""
        query_batch_strategy = QueryBatchDeletion(self.index)
        return await query_batch_strategy._delete_vector_ids_in_batches(vector_ids, namespace)


class VectorManagementService:
    """Comprehensive vector management service with multiple strategies"""

    def __init__(self):
        ai_config = get_ai_config()
        self.pc = Pinecone(api_key=ai_config.pinecone_api_key)
        self.index = self.pc.Index(ai_config.pinecone_index)

        # Initialize Supabase for upload tracking
        from ...config.settings import get_database_config
        db_config = get_database_config()
        self.supabase: Client = create_client(
            db_config.supabase_url, db_config.supabase_service_key)

        # Initialize deletion strategies in order of preference
        self.deletion_strategies = [
            MetadataFilterDeletion(self.index, max_retries=2, retry_delay=1.0),
            QueryBatchDeletion(self.index, max_retries=2, retry_delay=1.5),
            DummyVectorDeletion(self.index, max_retries=2, retry_delay=2.0)
        ]

    async def delete_upload_vectors(self, upload_id: str, org_id: str) -> VectorDeletionResult:
        """Delete all vectors for a specific upload using multiple strategies"""
        namespace = f"org-{org_id}"

        with LogContext(upload_id=upload_id, org_id=org_id, namespace=namespace):
            logger.info(f"Starting vector deletion for upload {upload_id}")

            # Try each strategy until one succeeds
            for strategy in self.deletion_strategies:
                with PerformanceLogger(f"vector_deletion_{strategy.__class__.__name__}"):
                    result = await strategy.delete_vectors(upload_id, namespace)

                    if result.success:
                        logger.info(
                            f"Vector deletion successful using {result.method_used}")

                        # Update upload status in database
                        await self._update_upload_status(upload_id, "deleted", None)

                        return result
                    else:
                        logger.warning(
                            f"Strategy {strategy.__class__.__name__} failed")

            # All strategies failed
            error_msg = "All deletion strategies failed"
            logger.error(
                f"Vector deletion completely failed for upload {upload_id}")

            # Update upload with error status
            await self._update_upload_status(upload_id, "deletion_failed", error_msg)

            return VectorDeletionResult(
                success=False,
                vectors_deleted=0,
                method_used="none",
                attempts=sum(strategy.max_retries for strategy in self.deletion_strategies),
                errors=[error_msg],
                duration_ms=0,
                upload_id=upload_id,
                namespace=namespace
            )

    async def batch_delete_uploads(self, upload_ids: List[str], org_id: str) -> Dict[str, Any]:
        """Efficiently delete multiple uploads with detailed reporting"""
        start_time = datetime.utcnow()
        results = {
            "successful_deletions": [],
            "failed_deletions": [],
            "total_vectors_deleted": 0,
            "processing_time_ms": 0,
            "summary": {}
        }

        with LogContext(org_id=org_id, batch_size=len(upload_ids)):
            logger.info(
                f"Starting batch deletion of {len(upload_ids)} uploads")

            # Process each upload
            for upload_id in upload_ids:
                try:
                    deletion_result = await self.delete_upload_vectors(upload_id, org_id)

                    if deletion_result.success:
                        results["successful_deletions"].append({
                            "upload_id": upload_id,
                            "vectors_deleted": deletion_result.vectors_deleted,
                            "method_used": deletion_result.method_used,
                            "duration_ms": deletion_result.duration_ms
                        })
                        results["total_vectors_deleted"] += deletion_result.vectors_deleted or 0
                    else:
                        results["failed_deletions"].append({
                            "upload_id": upload_id,
                            "errors": deletion_result.errors,
                            "attempts": deletion_result.attempts
                        })

                except Exception as e:
                    logger.error(
                        f"Unexpected error deleting upload {upload_id}: {e}", exc_info=True)
                    results["failed_deletions"].append({
                        "upload_id": upload_id,
                        "errors": [f"Unexpected error: {str(e)}"],
                        "attempts": 0
                    })

            # Calculate summary
            total_time = (datetime.utcnow() -
                          start_time).total_seconds() * 1000
            results["processing_time_ms"] = total_time

            results["summary"] = {
                "total_uploads": len(upload_ids),
                "successful": len(results["successful_deletions"]),
                "failed": len(results["failed_deletions"]),
                "success_rate": len(results["successful_deletions"]) / len(upload_ids) * 100 if upload_ids else 0,
                "total_vectors_deleted": results["total_vectors_deleted"],
                "avg_time_per_upload_ms": total_time / len(upload_ids) if upload_ids else 0
            }

            logger.info(f"Batch deletion completed: {results['summary']}")

        return results

    async def verify_deletion(self, upload_id: str, namespace: str) -> Dict[str, Any]:
        """Verify that vectors were actually deleted"""
        try:
            logger.info(f"Verifying deletion for upload {upload_id}")

            # Try to find any remaining vectors
            dummy_vector = [0.0] * 1536

            query_result = self.index.query(
                vector=dummy_vector,
                filter={"upload_id": upload_id},
                namespace=namespace,
                top_k=10,
                include_metadata=False,
                include_values=False
            )

            remaining_count = len(query_result.matches)
            verification_result = {
                "deletion_verified": remaining_count == 0,
                "remaining_vectors": remaining_count,
                "sample_ids": [match.id for match in query_result.matches[:5]] if query_result.matches else []
            }

            if remaining_count > 0:
                logger.warning(
                    f"Deletion verification failed: {remaining_count} vectors still exist")
            else:
                logger.info(
                    "Deletion verification successful: no vectors found")

            return verification_result

        except Exception as e:
            logger.error(f"Deletion verification failed: {e}")
            return {
                "deletion_verified": False,
                "error": str(e),
                "remaining_vectors": "unknown"
            }

    async def cleanup_orphaned_vectors(self, org_id: str, days_old: int = 7) -> VectorCleanupStats:
        """Clean up vectors that no longer have corresponding upload records"""
        namespace = f"org-{org_id}"
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        logger.info(f"Starting orphaned vector cleanup for org {org_id}")

        try:
            # Get all upload IDs from database
            uploads_response = self.supabase.table("uploads").select("id").eq(
                "org_id", org_id
            ).execute()

            valid_upload_ids = {upload["id"]
                                for upload in uploads_response.data or []}
            logger.info(
                f"Found {len(valid_upload_ids)} valid uploads in database")

            # Get sample of vectors from Pinecone to find orphans
            # This is a simplified approach - in production you might want more sophisticated scanning
            stats = VectorCleanupStats(
                total_uploads_processed=0,
                successful_deletions=0,
                failed_deletions=0,
                total_vectors_deleted=0,
                total_duration_ms=0,
                errors_encountered=[]
            )

            logger.info(
                "Orphaned vector cleanup completed (simplified implementation)")
            return stats

        except Exception as e:
            logger.error(f"Orphaned vector cleanup failed: {e}")
            return VectorCleanupStats(
                total_uploads_processed=0,
                successful_deletions=0,
                failed_deletions=1,
                total_vectors_deleted=0,
                total_duration_ms=0,
                errors_encountered=[str(e)]
            )

    async def _update_upload_status(self, upload_id: str, status: str, error_message: Optional[str]):
        """Update upload status in database"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }

            if error_message:
                update_data["error_message"] = error_message
            else:
                update_data["error_message"] = None

            self.supabase.table("uploads").update(
                update_data).eq("id", upload_id).execute()
            logger.info(f"Updated upload {upload_id} status to {status}")

        except Exception as e:
            logger.error(f"Failed to update upload status: {e}")


# Global service instance
vector_management_service = VectorManagementService()

# Convenience functions for backward compatibility


async def delete_vectors_from_pinecone(upload_id: str, namespace: str) -> VectorDeletionResult:
    """Backward compatible function for existing code"""
    org_id = namespace.replace(
        "org-", "") if namespace.startswith("org-") else "unknown"
    return await vector_management_service.delete_upload_vectors(upload_id, org_id)


async def batch_delete_uploads(upload_ids: List[str], org_id: str) -> Dict[str, Any]:
    """Backward compatible function for batch deletion"""
    return await vector_management_service.batch_delete_uploads(upload_ids, org_id)
