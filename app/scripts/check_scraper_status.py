"""
Check E-Commerce Scraper Status

This checks if the e-commerce scraper is properly available and configured.
"""

import os
import sys

# Get the backend directory (app's parent)
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add backend to path
sys.path.insert(0, backend_dir)

# IMPORTANT: Add venv's site-packages to Python path so it can find playwright
# when importing app modules that depend on it
venv_site_packages = os.path.join(
    backend_dir, ".venv", "lib", "python3.13", "site-packages"
)
if os.path.exists(venv_site_packages):
    sys.path.insert(0, venv_site_packages)

try:
    from app.services.scraping.cached_web_scraper import CachedWebScraper
    from app.services.scraping.ingestion_worker import (
        ECOMMERCE_SCRAPER_AVAILABLE,
        PLAYWRIGHT_AVAILABLE,
        is_ecommerce_url,
    )
    from app.services.scraping.scraping_cache_service import ScrapingCacheConfig

    print("=" * 80)
    print("🔍 E-COMMERCE SCRAPER STATUS CHECK")
    print("=" * 80)
    print()

    # Check 1: Scraper availability
    print("1. E-Commerce Scraper Availability:")
    if ECOMMERCE_SCRAPER_AVAILABLE:
        print("   ✅ E-Commerce Scraper: AVAILABLE")
    else:
        print("   ❌ E-Commerce Scraper: NOT AVAILABLE")

    if PLAYWRIGHT_AVAILABLE:
        print("   ✅ Playwright Scraper: AVAILABLE")
    else:
        print("   ❌ Playwright Scraper: NOT AVAILABLE")

    # Check 2: Cache configuration
    print("\n2. Cache Configuration:")
    config = ScrapingCacheConfig()
    if config.url_content_ttl == 0:
        print("   ✅ URL Cache TTL: DISABLED (0 seconds)")
    else:
        print(f"   ⚠️  URL Cache TTL: {config.url_content_ttl} seconds")
        print("      (Should be 0 for e-commerce URLs)")

    # Check 3: URL detection
    print("\n3. E-Commerce URL Detection:")
    test_urls = [
        "https://ambassadorscentworks.com/collections/talisman-series",
        "https://cakenbake.ae/",
        "https://example.com/products/test",
        "https://example.com/about",
    ]

    for url in test_urls:
        is_ecommerce = is_ecommerce_url(url)
        status = "✅" if is_ecommerce else "❌"
        print(
            f"   {status} {url[:60]}... → {'E-commerce' if is_ecommerce else 'Regular'} URL"
        )

    # Check 4: Cached Web Scraper
    print("\n4. Cached Web Scraper Cache Bypass:")
    try:
        scraper = CachedWebScraper()
        print("   ✅ CachedWebScraper initialized")
        print(f"   Cache enabled: {scraper.caching_enabled}")

        # Check the source code for e-commerce bypass
        import inspect

        source = inspect.getsource(scraper.scrape_url_text)
        has_ecommerce_bypass = "is_ecommerce" in source or "shopify" in source
        if has_ecommerce_bypass:
            print("   ✅ E-commerce cache bypass: ACTIVE")
        else:
            print("   ❌ E-commerce cache bypass: NOT FOUND in code")
    except Exception as e:
        print(f"   ❌ Error checking CachedWebScraper: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_good = ECOMMERCE_SCRAPER_AVAILABLE and config.url_content_ttl == 0

    if all_good:
        print("✅ All systems configured correctly!")
        print("\nIf you're still seeing uncleaned chunks:")
        print("1. Restart your backend server")
        print("2. Re-upload the URLs through dashboard")
        print("3. Check logs for '[E-commerce]' messages")
    else:
        print("❌ Issues detected:")
        if not ECOMMERCE_SCRAPER_AVAILABLE:
            print("   - E-commerce scraper not available")
        if config.url_content_ttl != 0:
            print("   - URL cache still enabled")

        print("\nTo fix:")
        print("1. Make sure backend code changes are deployed")
        print(
            "2. Restart backend: pkill -f 'python.*app.main' && cd backend && python -m app.main"
        )
        print("3. Re-upload URLs")

except Exception as e:
    print(f"❌ Error running status check: {e}")
    import traceback

    traceback.print_exc()
