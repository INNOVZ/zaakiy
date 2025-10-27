"""
Diagnostic: Test the Complete Scraping Flow

This will help identify where the e-commerce scraper is failing.

Run: python -m app.scripts.diagnose_scraping_flow
"""

import asyncio
import os
import sys

# Setup path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
venv_site_packages = os.path.join(
    backend_dir, ".venv", "lib", "python3.13", "site-packages"
)
if os.path.exists(venv_site_packages):
    sys.path.insert(0, venv_site_packages)

from app.services.scraping.ingestion_worker import (
    ECOMMERCE_SCRAPER_AVAILABLE,
    PLAYWRIGHT_AVAILABLE,
    is_ecommerce_url,
    smart_scrape_url,
)


async def test_complete_flow():
    """Test the complete scraping flow"""

    print("=" * 80)
    print("🔍 DIAGNOSTIC: Complete Scraping Flow Test")
    print("=" * 80)
    print()

    # Test URL
    test_url = "https://ambassadorscentworks.com/collections/new-arrivals"

    # Step 1: Check availability
    print("Step 1: Check Scraper Availability")
    print(
        f"  E-Commerce Scraper: {'✅ AVAILABLE' if ECOMMERCE_SCRAPER_AVAILABLE else '❌ NOT AVAILABLE'}"
    )
    print(
        f"  Playwright Scraper: {'✅ AVAILABLE' if PLAYWRIGHT_AVAILABLE else '❌ NOT AVAILABLE'}"
    )
    print()

    # Step 2: Check URL detection
    print("Step 2: Check E-Commerce Detection")
    is_ecommerce = is_ecommerce_url(test_url)
    print(f"  URL: {test_url}")
    print(f"  Detected as e-commerce: {'✅ YES' if is_ecommerce else '❌ NO'}")
    print()

    if not is_ecommerce:
        print("❌ PROBLEM: URL not detected as e-commerce!")
        print("   This means e-commerce scraper will NOT be used.")
        return

    # Step 3: Try to scrape
    if not ECOMMERCE_SCRAPER_AVAILABLE:
        print("❌ PROBLEM: E-Commerce Scraper not available!")
        print("   Cannot test scraping.")
        return

    print("Step 3: Attempt Scraping")
    print(f"  Calling smart_scrape_url('{test_url}')...")
    print()

    try:
        result = await smart_scrape_url(test_url)

        print("✅ Scraping completed!")
        print(f"  Result type: {type(result)}")
        print(f"  Result length: {len(result)} characters")
        print()

        # Check if it's structured
        is_structured = any(
            marker in result
            for marker in [
                "PRODUCTS (",
                "CONTACT INFORMATION:",
                "COLLECTION:",
                "has_products:",
            ]
        )

        print("Step 4: Analyze Result")
        if is_structured:
            print("  ✅ Result appears to be STRUCTURED")
        else:
            print("  ❌ Result is UNSTRUCTURED (no e-commerce formatting)")
            print("     This means the e-commerce scraper didn't work!")

        print()
        print("Step 5: Show First 500 Characters")
        print("-" * 80)
        print(result[:500])
        print("-" * 80)

        # Check for UI noise
        has_noise = any(
            noise in result
            for noise in [
                "Skip to content",
                "Your cart is empty",
                "image/svg+xml",
                "Sign In Email Password",
            ]
        )

        print()
        if has_noise:
            print("❌ PROBLEM FOUND: Result contains UI noise!")
            print(
                "   Examples: 'Skip to content', 'Your cart is empty', 'image/svg+xml'"
            )
        else:
            print("✅ Result is clean (no UI noise detected)")

    except Exception as e:
        print(f"❌ ERROR during scraping: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_complete_flow())
