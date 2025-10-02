#!/usr/bin/env python3
"""
Data migration script for Zentria backend

This script helps migrate data between environments or perform
database schema updates and data transformations.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.storage.supabase_client import client
from services.shared.client_manager import client_manager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataMigrator:
    """Data migration utility for Zentria backend"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.migration_log = []
        self.start_time = datetime.utcnow()
    
    def log_migration(self, operation: str, details: str, status: str = "info"):
        """Log migration operation"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "details": details,
            "status": status
        }
        self.migration_log.append(entry)
        
        status_emoji = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
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
            backup_filename = f"backup_{table_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join("backups", backup_filename)
            
            # Ensure backups directory exists
            os.makedirs("backups", exist_ok=True)
            
            if not self.dry_run:
                import json
                with open(backup_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            self.log_migration("Backup", f"Backed up {len(data)} records to {backup_path}", "success")
            return True
            
        except Exception as e:
            self.log_migration("Backup", f"Error backing up {table_name}: {str(e)}", "error")
            return False
    
    async def migrate_users_table(self) -> bool:
        """Migrate users table structure"""
        try:
            self.log_migration("Migration", "Starting users table migration")
            
            # Get all users
            response = await client.get("/users", params={"select": "*"})
            
            if response.status_code != 200:
                self.log_migration("Migration", "Failed to read users table", "error")
                return False
            
            users = response.json()
            migrated_count = 0
            
            for user in users:
                # Example migration: add missing fields
                updates = {}
                
                # Add created_at if missing
                if not user.get("created_at"):
                    updates["created_at"] = datetime.utcnow().isoformat()
                
                # Add updated_at if missing
                if not user.get("updated_at"):
                    updates["updated_at"] = datetime.utcnow().isoformat()
                
                # Add default role if missing
                if not user.get("role"):
                    updates["role"] = "user"
                
                # Update user if changes needed
                if updates:
                    if not self.dry_run:
                        update_response = await client.patch(
                            f"/users?id=eq.{user['id']}",
                            json=updates
                        )
                        
                        if update_response.status_code in [200, 204]:
                            migrated_count += 1
                        else:
                            self.log_migration("Migration", f"Failed to update user {user['id']}", "warning")
                    else:
                        migrated_count += 1
                        self.log_migration("Migration", f"Would update user {user['id']} with {updates}")
            
            self.log_migration("Migration", f"Migrated {migrated_count} users", "success")
            return True
            
        except Exception as e:
            self.log_migration("Migration", f"Error migrating users: {str(e)}", "error")
            return False
    
    async def migrate_organizations_table(self) -> bool:
        """Migrate organizations table structure"""
        try:
            self.log_migration("Migration", "Starting organizations table migration")
            
            # Get all organizations
            response = await client.get("/organizations", params={"select": "*"})
            
            if response.status_code != 200:
                self.log_migration("Migration", "Failed to read organizations table", "error")
                return False
            
            orgs = response.json()
            migrated_count = 0
            
            for org in orgs:
                updates = {}
                
                # Add created_at if missing
                if not org.get("created_at"):
                    updates["created_at"] = datetime.utcnow().isoformat()
                
                # Add updated_at if missing
                if not org.get("updated_at"):
                    updates["updated_at"] = datetime.utcnow().isoformat()
                
                # Add default plan_id if missing
                if not org.get("plan_id"):
                    updates["plan_id"] = "free"
                
                # Update organization if changes needed
                if updates:
                    if not self.dry_run:
                        update_response = await client.patch(
                            f"/organizations?id=eq.{org['id']}",
                            json=updates
                        )
                        
                        if update_response.status_code in [200, 204]:
                            migrated_count += 1
                        else:
                            self.log_migration("Migration", f"Failed to update org {org['id']}", "warning")
                    else:
                        migrated_count += 1
                        self.log_migration("Migration", f"Would update org {org['id']} with {updates}")
            
            self.log_migration("Migration", f"Migrated {migrated_count} organizations", "success")
            return True
            
        except Exception as e:
            self.log_migration("Migration", f"Error migrating organizations: {str(e)}", "error")
            return False
    
    async def cleanup_orphaned_data(self) -> bool:
        """Clean up orphaned data"""
        try:
            self.log_migration("Cleanup", "Starting orphaned data cleanup")
            
            # Example: Find users without organizations
            response = await client.get("/users", params={
                "select": "id,email,org_id",
                "org_id": "is.null"
            })
            
            if response.status_code == 200:
                orphaned_users = response.json()
                
                if orphaned_users:
                    self.log_migration("Cleanup", f"Found {len(orphaned_users)} users without organizations", "warning")
                    
                    if not self.dry_run:
                        # Create default organization for orphaned users
                        for user in orphaned_users:
                            # Create organization
                            org_data = {
                                "name": f"Organization for {user['email']}",
                                "email": user['email'],
                                "plan_id": "free"
                            }
                            
                            org_response = await client.post("/organizations", json=org_data)
                            
                            if org_response.status_code == 201:
                                org_id = org_response.json()["id"]
                                
                                # Update user with org_id
                                await client.patch(
                                    f"/users?id=eq.{user['id']}",
                                    json={"org_id": org_id}
                                )
                                
                                self.log_migration("Cleanup", f"Created org for user {user['email']}", "success")
                else:
                    self.log_migration("Cleanup", "No orphaned users found", "success")
            
            return True
            
        except Exception as e:
            self.log_migration("Cleanup", f"Error cleaning up orphaned data: {str(e)}", "error")
            return False
    
    async def run_migration(self, migration_type: str = "all") -> bool:
        """Run the specified migration"""
        self.log_migration("Migration", f"Starting {migration_type} migration (dry_run={self.dry_run})")
        
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
            self.log_migration("Migration", f"All {total_count} migrations completed successfully", "success")
            return True
        else:
            self.log_migration("Migration", f"{success_count}/{total_count} migrations succeeded", "warning")
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
            
            status_emoji = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌"
            }
            
            print(f"{status_emoji.get(status, 'ℹ️')} [{timestamp}] {operation}: {details}")
        
        print(f"\nTotal operations: {len(self.migration_log)}")
        print(f"Duration: {(datetime.utcnow() - self.start_time).total_seconds():.2f} seconds")


async def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Zentria Backend Data Migration")
    parser.add_argument("--type", choices=["all", "users", "organizations", "cleanup"], 
                       default="all", help="Migration type to run")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Run in dry-run mode (default)")
    parser.add_argument("--execute", action="store_true",
                       help="Execute the migration (overrides dry-run)")
    
    args = parser.parse_args()
    
    # Determine if this is a dry run
    dry_run = args.dry_run and not args.execute
    
    print("🔄 Zentria Backend Data Migration")
    print("=" * 50)
    print(f"Migration type: {args.type}")
    print(f"Dry run: {dry_run}")
    print(f"Started at: {datetime.utcnow().isoformat()}")
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
        print("   Use --execute to perform actual migration")
    
    migrator = DataMigrator(dry_run=dry_run)
    
    try:
        success = await migrator.run_migration(args.type)
        migrator.print_migration_log()
        
        if success:
            print(f"\n🎉 Migration completed successfully!")
            if dry_run:
                print("   Run with --execute to apply changes")
            sys.exit(0)
        else:
            print(f"\n⚠️  Migration completed with warnings/errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        migrator.print_migration_log()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
