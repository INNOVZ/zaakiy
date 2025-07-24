import os
from dotenv import load_dotenv

print("🔍 Debugging environment variables...")
load_dotenv()

variables = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
             "SUPABASE_ANON_KEY", "SUPABASE_JWT_SECRET"]

missing_vars = []
for var in variables:
    value = os.getenv(var)
    if value is None:
        print(f"❌ {var}: None (MISSING)")
        missing_vars.append(var)
    else:
        print(f"✅ {var}: {value[:20]}...")

print(f"\nMissing variables: {missing_vars}")
