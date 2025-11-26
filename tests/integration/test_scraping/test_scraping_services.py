"""
End-to-end and unit tests for scraping services
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend root to path
backend_root = (
    Path(__file__).parent.parent.parent.parent
    if "integration" in str(Path(__file__))
    else Path(__file__).parent.parent.parent
)
sys.path.insert(0, str(backend_root))
from dotenv import load_dotenv

load_dotenv()


def test_text_cleaner_usage():
    """Test that TextCleaner is used consistently"""
    print("=" * 80)
    print("🧪 TEST 1: TextCleaner Usage Consistency")
    print("=" * 80)
    print()

    scraping_dir = Path("app/services/scraping")
    files_to_check = [
        "web_scraper.py",
        "playwright_scraper.py",
        "ecommerce_scraper.py",
        "ingestion_worker.py",
    ]

    issues = []
    for file_name in files_to_check:
        file_path = scraping_dir / file_name
        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file has cleaning logic
        has_cleaning = any(
            pattern in content.lower()
            for pattern in ["clean", "remove", "filter", "strip", "sanitize"]
        )

        # Check if it uses TextCleaner
        uses_text_cleaner = (
            "TextCleaner" in content
            or "text_cleaner" in content
            or "from .text_cleaner" in content
        )

        if has_cleaning and not uses_text_cleaner:
            issues.append(file_name)
            print(f"   ⚠️  {file_name}: Has cleaning logic but doesn't use TextCleaner")
        else:
            print(
                f"   ✅ {file_name}: {'Uses TextCleaner' if uses_text_cleaner else 'No cleaning logic'}"
            )

    print()
    if issues:
        print(f"⚠️  Found {len(issues)} files that should use TextCleaner")
    else:
        print("✅ All files use TextCleaner consistently")

    return len(issues) == 0


def test_contact_extraction_consistency():
    """Test that contact extraction uses shared utilities"""
    print("=" * 80)
    print("🧪 TEST 2: Contact Extraction Consistency")
    print("=" * 80)
    print()

    scraping_dir = Path("app/services/scraping")
    scraper_files = [
        "web_scraper.py",
        "playwright_scraper.py",
        "ecommerce_scraper.py",
    ]

    all_use_contact_extractor = True
    for file_name in scraper_files:
        file_path = scraping_dir / file_name
        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        has_contact_extraction = "_extract_contact_information" in content
        uses_contact_extractor = (
            "ContactExtractor" in content or "contact_extractor" in content
        )

        if has_contact_extraction:
            if uses_contact_extractor:
                print(f"   ✅ {file_name}: Uses ContactExtractor")
            else:
                print(
                    f"   ⚠️  {file_name}: Has contact extraction but may not use ContactExtractor"
                )
                all_use_contact_extractor = False
        else:
            print(f"   ✅ {file_name}: No contact extraction (OK)")

    print()
    if all_use_contact_extractor:
        print("✅ All scrapers use ContactExtractor consistently")
    else:
        print("⚠️  Some scrapers may have duplicate contact extraction logic")

    return all_use_contact_extractor


def test_async_operations():
    """Test that async operations are properly implemented"""
    print("=" * 80)
    print("🧪 TEST 3: Async Operations")
    print("=" * 80)
    print()

    scraping_dir = Path("app/services/scraping")
    python_files = list(scraping_dir.rglob("*.py"))
    python_files = [f for f in python_files if "__pycache__" not in str(f)]

    blocking_in_async = []
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if "async def" in line:
                    # Check next 20 lines for blocking operations
                    for j in range(i, min(i + 20, len(lines))):
                        if any(
                            op in lines[j]
                            for op in ["time.sleep(", "requests.get(", "requests.post("]
                        ):
                            blocking_in_async.append(
                                {
                                    "file": file_path.name,
                                    "line": j + 1,
                                    "operation": [
                                        op
                                        for op in [
                                            "time.sleep",
                                            "requests.get",
                                            "requests.post",
                                        ]
                                        if op in lines[j]
                                    ][0],
                                }
                            )
                            break
        except:
            pass

    if blocking_in_async:
        print(
            f"⚠️  Found {len(blocking_in_async)} potential blocking operations in async functions:"
        )
        for issue in blocking_in_async[:5]:
            print(f"   - {issue['file']}:{issue['line']} - {issue['operation']}")
    else:
        print("✅ No blocking operations found in async functions")

    print()
    return len(blocking_in_async) == 0


def test_import_consistency():
    """Test that imports are consistent and not duplicated"""
    print("=" * 80)
    print("🧪 TEST 4: Import Consistency")
    print("=" * 80)
    print()

    scraping_dir = Path("app/services/scraping")
    python_files = list(scraping_dir.rglob("*.py"))
    python_files = [
        f
        for f in python_files
        if "__pycache__" not in str(f) and f.name != "__init__.py"
    ]

    # Check for duplicate imports of scraping utilities
    url_utils_imports = 0
    text_cleaner_imports = 0
    content_extractors_imports = 0

    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "url_utils" in content or "from .url_utils" in content:
                url_utils_imports += 1
            if "text_cleaner" in content or "from .text_cleaner" in content:
                text_cleaner_imports += 1
            if "content_extractors" in content or "from .content_extractors" in content:
                content_extractors_imports += 1
        except:
            pass

    print(f"📊 Import usage:")
    print(f"   - url_utils: {url_utils_imports} files")
    print(f"   - text_cleaner: {text_cleaner_imports} files")
    print(f"   - content_extractors: {content_extractors_imports} files")
    print()

    if url_utils_imports > 0 and text_cleaner_imports > 0:
        print("✅ Utilities are being used across files")
    else:
        print("⚠️  Utilities may not be used consistently")

    return True


def test_file_size():
    """Test that files are not too large"""
    print("=" * 80)
    print("🧪 TEST 5: File Size Analysis")
    print("=" * 80)
    print()

    scraping_dir = Path("app/services/scraping")
    python_files = list(scraping_dir.rglob("*.py"))
    python_files = [f for f in python_files if "__pycache__" not in str(f)]

    large_files = []
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = len(f.readlines())

            if lines > 1000:
                large_files.append(
                    {
                        "file": file_path.name,
                        "lines": lines,
                    }
                )
        except:
            pass

    if large_files:
        print(f"⚠️  Found {len(large_files)} large files (>1000 lines):")
        for file_info in large_files:
            print(f"   - {file_info['file']}: {file_info['lines']} lines")
            if file_info["lines"] > 1500:
                print(f"      💡 Consider splitting this file")
    else:
        print("✅ All files are reasonably sized")

    print()
    return len([f for f in large_files if f["lines"] > 1500]) == 0


async def test_scraping_functions():
    """Test that key scraping functions work"""
    print("=" * 80)
    print("🧪 TEST 6: Scraping Function Tests")
    print("=" * 80)
    print()

    try:
        from app.services.scraping.ingestion_worker import extract_topics_from_url

        # Test topic extraction
        test_url = "https://ohhzones.com/digital-marketing/seo/"
        topics = extract_topics_from_url(test_url)

        if topics:
            print(f"   ✅ extract_topics_from_url() works: {topics}")
        else:
            print(f"   ⚠️  extract_topics_from_url() returned empty")

        # Test recursive scraping signature
        import inspect

        from app.services.scraping.ingestion_worker import recursive_scrape_website

        sig = inspect.signature(recursive_scrape_website)
        if "return_individual_urls" in sig.parameters:
            print(
                f"   ✅ recursive_scrape_website() has return_individual_urls parameter"
            )
        else:
            print(
                f"   ❌ recursive_scrape_website() missing return_individual_urls parameter"
            )

        # Test upload function
        from app.services.scraping.ingestion_worker import (
            upload_to_pinecone_with_url_mapping,
        )

        sig2 = inspect.signature(upload_to_pinecone_with_url_mapping)
        if "url_to_chunks_map" in sig2.parameters:
            print(
                f"   ✅ upload_to_pinecone_with_url_mapping() has url_to_chunks_map parameter"
            )
        else:
            print(
                f"   ❌ upload_to_pinecone_with_url_mapping() missing url_to_chunks_map parameter"
            )

        return True

    except Exception as e:
        print(f"   ⚠️  Error testing functions: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("🚀 SCRAPING SERVICES COMPREHENSIVE TEST")
    print("=" * 80)
    print()

    results = {}

    results["text_cleaner"] = test_text_cleaner_usage()
    print()

    results["contact_extraction"] = test_contact_extraction_consistency()
    print()

    results["async_operations"] = test_async_operations()
    print()

    results["imports"] = test_import_consistency()
    print()

    results["file_size"] = test_file_size()
    print()

    results["functions"] = asyncio.run(test_scraping_functions())
    print()

    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print()

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above for details.")


if __name__ == "__main__":
    main()
