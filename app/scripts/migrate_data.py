#!/usr/bin/env python3
"""
Data migration script for Zentria backend

This script helps migrate data between environments or perform
database schema updates and data transformations.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.shared.client_manager import client_manager
from services.storage.supabase_client import client

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataMigrator:
    """Data migration utility for Zentria backend"""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.migration_log = []
        self.start_time = datetime.now(timezone.utc)

    def log_migration(self, operation: str, details: str, status: str = "info"):
        """Log migration operation"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "details": details,
            "status": status,
        }
        self.migration_log.append(entry)

        status_emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}

        print(f"{status_emoji.get(status, 'ℹ️')} {operation}: {details}")

    async def backup_table(self, table_name: str) -> bool:
        """Create a backup of a table"""
        try:
            self.log_migration("Backup", f"Starting backup of {table_name}")

            # Get all data from table
            response = await client.get(f"/{table_name}", params={"select": "*"})

            if response.status_code != 200:
                self.log_migration("Backup", f"Failed to read {table_name}", "error")
                return False

            data = response.json()

            # Create backup file
            backup_filename = f"backup_{table_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join("backups", backup_filename)

            # Ensure backups directory exists
            os.makedirs("backups", exist_ok=True)

            if not self.dry_run:
                import json

                import aiofiles

                async with aiofiles.open(backup_path, "w") as f:
                    await f.write(json.dumps(data, indent=2, default=str))

            self.log_migration(
                "Backup", f"Backed up {len(data)} records to {backup_path}", "success"
            )
            return True

        except Exception as e:
            self.log_migration(
                "Backup", f"Error backing up {table_name}: {str(e)}", "error"
            )
            return False

    async def migrate_users_table(self) -> bool:
        """Migrate users table structure"""
        try:
            self.log_migration("Migration", "Starting users table migration")

            users = await self._fetch_users()
            if users is None:
                return False

            migrated_count = await self._migrate_users_data(users)
            self.log_migration(
                "Migration", f"Migrated {migrated_count} users", "success"
            )
            return True

        except Exception as e:
            self.log_migration("Migration", f"Error migrating users: {str(e)}", "error")
            return False

    async def _fetch_users(self) -> list:
        """Fetch all users from the database"""
        response = await client.get("/users", params={"select": "*"})

        if response.status_code != 200:
            self.log_migration("Migration", "Failed to read users table", "error")
            return None

        return response.json()

    async def _migrate_users_data(self, users: list) -> int:
        """Migrate user data and return count of migrated users"""
        migrated_count = 0

        for user in users:
            updates = self._build_user_updates(user)

            if updates:
                success = await self._update_user_record(user["id"], updates)
                if success:
                    migrated_count += 1

        return migrated_count

    def _build_user_updates(self, user: dict) -> dict:
        """Build update dictionary for a user record"""
        updates = {}

        # Add created_at if missing
        if not user.get("created_at"):
            updates["created_at"] = datetime.now(timezone.utc).isoformat()

        # Add updated_at if missing
        if not user.get("updated_at"):
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Add default role if missing
        if not user.get("role"):
            updates["role"] = "user"

        return updates

    async def _update_user_record(self, user_id: str, updates: dict) -> bool:
        """Update a single user record"""
        if not self.dry_run:
            update_response = await client.patch(
                f"/users?id=eq.{user_id}", json=updates
            )

            if update_response.status_code in [200, 204]:
                return True
            else:
                self.log_migration(
                    "Migration", f"Failed to update user {user_id}", "warning"
                )
                return False
        else:
            self.log_migration(
                "Migration", f"Would update user {user_id} with {updates}"
            )
            return True

    async def migrate_organizations_table(self) -> bool:
        """Migrate organizations table structure"""
        try:
            self.log_migration("Migration", "Starting organizations table migration")

            orgs = await self._fetch_organizations()
            if orgs is None:
                return False

            migrated_count = await self._migrate_organizations_data(orgs)
            self.log_migration(
                "Migration", f"Migrated {migrated_count} organizations", "success"
            )
            return True

        except Exception as e:
            self.log_migration(
                "Migration", f"Error migrating organizations: {str(e)}", "error"
            )
            return False

    async def _fetch_organizations(self) -> list:
        """Fetch all organizations from the database"""
        response = await client.get("/organizations", params={"select": "*"})

        if response.status_code != 200:
            self.log_migration(
                "Migration", "Failed to read organizations table", "error"
            )
            return None

        return response.json()

    async def _migrate_organizations_data(self, orgs: list) -> int:
        """Migrate organization data and return count of migrated organizations"""
        migrated_count = 0

        for org in orgs:
            updates = self._build_organization_updates(org)

            if updates:
                success = await self._update_organization_record(org["id"], updates)
                if success:
                    migrated_count += 1

        return migrated_count

    def _build_organization_updates(self, org: dict) -> dict:
        """Build update dictionary for an organization record"""
        updates = {}

        # Add created_at if missing
        if not org.get("created_at"):
            updates["created_at"] = datetime.now(timezone.utc).isoformat()

        # Add updated_at if missing
        if not org.get("updated_at"):
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Add default plan_id if missing
        if not org.get("plan_id"):
            updates["plan_id"] = "free"

        return updates

    async def _update_organization_record(self, org_id: str, updates: dict) -> bool:
        """Update a single organization record"""
        if not self.dry_run:
            update_response = await client.patch(
                f"/organizations?id=eq.{org_id}", json=updates
            )

            if update_response.status_code in [200, 204]:
                return True
            else:
                self.log_migration(
                    "Migration", f"Failed to update org {org_id}", "warning"
                )
                return False
        else:
            self.log_migration("Migration", f"Would update org {org_id} with {updates}")
            return True

    async def cleanup_orphaned_data(self) -> bool:
        """Clean up orphaned data"""
        try:
            self.log_migration("Cleanup", "Starting orphaned data cleanup")

            orphaned_users = await self._fetch_orphaned_users()
            if orphaned_users is None:
                return False

            await self._process_orphaned_users(orphaned_users)
            return True

        except Exception as e:
            self.log_migration(
                "Cleanup", f"Error cleaning up orphaned data: {str(e)}", "error"
            )
            return False

    async def _fetch_orphaned_users(self) -> list:
        """Fetch users without organizations"""
        response = await client.get(
            "/users", params={"select": "id,email,org_id", "org_id": "is.null"}
        )

        if response.status_code != 200:
            self.log_migration("Cleanup", "Failed to fetch orphaned users", "error")
            return None

        return response.json()

    async def _process_orphaned_users(self, orphaned_users: list) -> None:
        """Process orphaned users by creating organizations for them"""
        if not orphaned_users:
            self.log_migration("Cleanup", "No orphaned users found", "success")
            return

        self.log_migration(
            "Cleanup",
            f"Found {len(orphaned_users)} users without organizations",
            "warning",
        )

        if not self.dry_run:
            await self._create_organizations_for_users(orphaned_users)

    async def _create_organizations_for_users(self, orphaned_users: list) -> None:
        """Create organizations for orphaned users"""
        for user in orphaned_users:
            success = await self._create_organization_for_user(user)
            if success:
                self.log_migration(
                    "Cleanup", f"Created org for user {user['email']}", "success"
                )

    async def _create_organization_for_user(self, user: dict) -> bool:
        """Create organization for a single user"""
        org_data = {
            "name": f"Organization for {user['email']}",
            "email": user["email"],
            "plan_id": "free",
        }

        org_response = await client.post("/organizations", json=org_data)

        if org_response.status_code == 201:
            org_id = org_response.json()["id"]

            # Update user with org_id
            await client.patch(f"/users?id=eq.{user['id']}", json={"org_id": org_id})

            return True

        return False

    async def run_migration(self, migration_type: str = "all") -> bool:
        """Run the specified migration"""
        self.log_migration(
            "Migration", f"Starting {migration_type} migration (dry_run={self.dry_run})"
        )

        # Create backups first
        tables_to_backup = ["users", "organizations", "conversations", "messages"]

        for table in tables_to_backup:
            await self.backup_table(table)

        # Run migrations
        migrations = []

        if migration_type in ["all", "users"]:
            migrations.append(self.migrate_users_table())

        if migration_type in ["all", "organizations"]:
            migrations.append(self.migrate_organizations_table())

        if migration_type in ["all", "cleanup"]:
            migrations.append(self.cleanup_orphaned_data())

        # Run all migrations
        results = await asyncio.gather(*migrations, return_exceptions=True)

        # Check results
        success_count = sum(1 for r in results if r is True)
        total_count = len(results)

        if success_count == total_count:
            self.log_migration(
                "Migration",
                f"All {total_count} migrations completed successfully",
                "success",
            )
            return True
        else:
            self.log_migration(
                "Migration",
                f"{success_count}/{total_count} migrations succeeded",
                "warning",
            )
            return False

    def print_migration_log(self):
        """Print migration log summary"""
        print("\n" + "=" * 60)
        print("📋 MIGRATION LOG SUMMARY")
        print("=" * 60)

        for entry in self.migration_log:
            timestamp = entry["timestamp"]
            operation = entry["operation"]
            details = entry["details"]
            status = entry["status"]

            status_emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}

            print(
                f"{status_emoji.get(status, 'ℹ️')} [{timestamp}] {operation}: {details}"
            )

        print(f"\nTotal operations: {len(self.migration_log)}")
        print(
            f"Duration: {(datetime.now(timezone.utc) - self.start_time).total_seconds():.2f} seconds"
        )


async def main():
    """Main migration function"""
    import argparse

    parser = argparse.ArgumentParser(description="Zentria Backend Data Migration")
    parser.add_argument(
        "--type",
        choices=["all", "users", "organizations", "cleanup"],
        default="all",
        help="Migration type to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the migration (overrides dry-run)",
    )

    args = parser.parse_args()

    # Determine if this is a dry run
    dry_run = args.dry_run and not args.execute

    print("🔄 Zentria Backend Data Migration")
    print("=" * 50)
    print(f"Migration type: {args.type}")
    print(f"Dry run: {dry_run}")
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")

    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
        print("   Use --execute to perform actual migration")

    migrator = DataMigrator(dry_run=dry_run)

    try:
        success = await migrator.run_migration(args.type)
        migrator.print_migration_log()

        if success:
            print("\n🎉 Migration completed successfully!")
            if dry_run:
                print("   Run with --execute to apply changes")
            sys.exit(0)
        else:
            print("\n⚠️  Migration completed with warnings/errors")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        migrator.print_migration_log()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
