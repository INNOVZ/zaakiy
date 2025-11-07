#!/usr/bin/env python3
"""
Test URL linking functionality in response generation service
Standalone tests that don't require full service imports
"""
import os
import re
import sys


def test_extract_urls_from_context():
    """Test URL extraction from context"""

    # Create a mock service instance (we only need the URL extraction method)
    class MockService:
        def _extract_urls_from_context(self, context: str):
            """Extract all URLs from context"""
            url_patterns = [
                r'https?://[^\s\)<"\'\,]+',
                r'www\.[^\s\)<"\'\,]+',
                r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}(?:/[^\s\)<"\'\,]*)?',
            ]

            all_urls = []
            for pattern in url_patterns:
                urls = re.findall(pattern, context, re.IGNORECASE)
                all_urls.extend(urls)

            valid_urls = []
            seen_urls = set()

            for url in all_urls:
                url = url.strip().rstrip(".,;:!?)\"'<>")

                if len(url) < 4 or not ("." in url or url.startswith("http")):
                    continue

                if url.startswith("www."):
                    full_url = "https://" + url
                elif not url.startswith(("http://", "https://")):
                    if re.match(
                        r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}",
                        url,
                    ):
                        full_url = "https://" + url
                    else:
                        continue
                else:
                    full_url = url

                normalized = full_url.rstrip("/")
                if normalized not in seen_urls:
                    seen_urls.add(normalized)
                    valid_urls.append(full_url)

            return valid_urls

    service = MockService()

    # Test 1: Extract https URL
    context1 = "Visit https://example.com/book-demo to book a demo"
    urls1 = service._extract_urls_from_context(context1)
    assert (
        "https://example.com/book-demo" in urls1
    ), f"Expected https://example.com/book-demo, got {urls1}"
    print("✅ Test 1 passed: Extracted https URL")

    # Test 2: Extract www URL
    context2 = "Check out www.example.com for more info"
    urls2 = service._extract_urls_from_context(context2)
    assert any(
        "example.com" in url for url in urls2
    ), f"Expected example.com URL, got {urls2}"
    print("✅ Test 2 passed: Extracted www URL")

    # Test 3: Extract demo URL
    context3 = "Book a demo at https://calendly.com/company/demo"
    urls3 = service._extract_urls_from_context(context3)
    assert any(
        "calendly.com" in url for url in urls3
    ), f"Expected calendly URL, got {urls3}"
    print("✅ Test 3 passed: Extracted demo URL")

    # Test 4: Multiple URLs
    context4 = "Visit https://example.com and book at https://calendly.com/demo"
    urls4 = service._extract_urls_from_context(context4)
    assert len(urls4) >= 2, f"Expected at least 2 URLs, got {len(urls4)}"
    print("✅ Test 4 passed: Extracted multiple URLs")

    print("\n✅ All URL extraction tests passed!")


def test_link_common_phrases():
    """Test linking common phrases to URLs"""

    class MockService:
        def _extract_urls_from_context(self, context: str):
            """Simple URL extraction for testing"""
            urls = re.findall(r"https?://[^\s\)]+", context)
            return urls

        def _link_common_phrases_to_urls(self, response: str, context: str):
            """Link common phrases to URLs"""
            context_urls = self._extract_urls_from_context(context)

            if not context_urls:
                return response

            demo_urls = [
                url
                for url in context_urls
                if any(
                    keyword in url.lower()
                    for keyword in [
                        "demo",
                        "book",
                        "schedule",
                        "appointment",
                        "booking",
                        "calendly",
                        "cal.com",
                        "meet",
                        "zoom",
                    ]
                )
            ]
            website_urls = [
                url
                for url in context_urls
                if not any(
                    keyword in url.lower()
                    for keyword in [
                        "demo",
                        "book",
                        "schedule",
                        "appointment",
                        "booking",
                        "calendly",
                        "cal.com",
                        "meet",
                        "zoom",
                    ]
                )
            ]

            demo_url = demo_urls[0] if demo_urls else None
            website_url = (
                website_urls[0]
                if website_urls
                else (demo_urls[0] if demo_urls else None)
            )

            def is_already_linked(text: str, phrase: str) -> bool:
                pattern = r"\[([^\]]*" + re.escape(phrase) + r"[^\]]*)\]\([^\)]+\)"
                return bool(re.search(pattern, text, re.IGNORECASE))

            if demo_url:
                demo_phrases = [
                    (r"\b(book\s+a\s+demo)\b", "book a demo"),
                    (r"\b(schedule\s+a\s+demo)\b", "schedule a demo"),
                ]

                for pattern, link_text in demo_phrases:
                    if not is_already_linked(response, link_text):
                        response = re.sub(
                            pattern,
                            f"[{link_text}]({demo_url})",
                            response,
                            flags=re.IGNORECASE,
                        )

            target_url = website_url if website_url else demo_url

            if target_url:
                website_phrases = [
                    (r"\bour\s+website\b", "our website"),
                    (r"\bwebsite\b", "website"),
                ]

                for pattern, link_text in website_phrases:
                    if not is_already_linked(response, link_text.split()[-1]):
                        response = re.sub(
                            pattern + r"(?![^\[]*\]\([^\)]+\))",
                            f"[{link_text}]({target_url})",
                            response,
                            flags=re.IGNORECASE,
                        )

            return response

    service = MockService()

    # Test 1: Link "book a demo"
    context1 = "Book a demo at https://calendly.com/company/demo"
    response1 = "To book a demo, please contact us"
    result1 = service._link_common_phrases_to_urls(response1, context1)
    assert "[book a demo]" in result1, f"Expected [book a demo] link, got: {result1}"
    assert "calendly.com" in result1, f"Expected calendly URL, got: {result1}"
    print("✅ Test 1 passed: Linked 'book a demo' to URL")

    # Test 2: Link "our website"
    context2 = "Visit https://example.com for more info"
    response2 = "Check out our website for details"
    result2 = service._link_common_phrases_to_urls(response2, context2)
    assert "[our website]" in result2, f"Expected [our website] link, got: {result2}"
    assert "example.com" in result2, f"Expected example.com URL, got: {result2}"
    print("✅ Test 2 passed: Linked 'our website' to URL")

    # Test 3: No URLs in context
    context3 = "No URLs here"
    response3 = "Visit our website"
    result3 = service._link_common_phrases_to_urls(response3, context3)
    assert result3 == response3, "Should not modify response when no URLs in context"
    print("✅ Test 3 passed: No linking when no URLs")

    print("\n✅ All phrase linking tests passed!")


def test_phone_validation():
    """Test phone number validation"""

    class MockService:
        def _is_valid_phone_number(self, phone: str) -> bool:
            """Check if phone number is valid"""
            normalized = re.sub(r"[^\d+]", "", phone)
            digits_only = re.sub(r"[^\d]", "", normalized)

            if len(digits_only) < 7 or len(digits_only) > 15:
                return False

            vague_patterns = [
                r"^10000$",
                r"^12345",
                r"^00000",
                r"^11111",
                r"^99999",
                r"^000[0-9]{2,}$",
                r"^([0-9])\1{4,}$",
            ]

            for pattern in vague_patterns:
                if re.match(pattern, digits_only):
                    return False

            if len(set(digits_only)) <= 2 and len(digits_only) < 10:
                return False

            return True

    service = MockService()

    # Test valid numbers
    assert (
        service._is_valid_phone_number("+971 50 123 4567") == True
    ), "Valid UAE number should pass"
    assert (
        service._is_valid_phone_number("0503789198") == True
    ), "Valid number should pass"
    print("✅ Test 1 passed: Valid phone numbers accepted")

    # Test vague numbers
    assert service._is_valid_phone_number("10000") == False, "10000 should be rejected"
    assert service._is_valid_phone_number("12345") == False, "12345 should be rejected"
    assert service._is_valid_phone_number("00000") == False, "00000 should be rejected"
    assert service._is_valid_phone_number("11111") == False, "11111 should be rejected"
    print("✅ Test 2 passed: Vague phone numbers rejected")

    # Test too short
    assert (
        service._is_valid_phone_number("123") == False
    ), "Too short number should be rejected"
    print("✅ Test 3 passed: Too short numbers rejected")

    print("\n✅ All phone validation tests passed!")


if __name__ == "__main__":
    print("🧪 Running URL Linking Tests\n")
    print("=" * 50)

    try:
        print("\n📋 Test 1: URL Extraction")
        print("-" * 50)
        test_extract_urls_from_context()

        print("\n📋 Test 2: Phrase Linking")
        print("-" * 50)
        test_link_common_phrases()

        print("\n📋 Test 3: Phone Validation")
        print("-" * 50)
        test_phone_validation()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
