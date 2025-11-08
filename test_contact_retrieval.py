"""
Test script to verify contact information retrieval
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.chat.contact_extractor import contact_extractor

# Test chunk with contact information (similar to Pinecone data)
TEST_CHUNK = """
{
  "unlimited Training Data": {"name": "unlimited Training Data", "included": true},
  "3 Zaakiy assistant": {"name": "3 Zaakiy assistant", "included": true},
  "Advanced analytics": {"name": "Advanced analytics", "included": true},
  "Email support": {"name": "Email support", "included": true},
  "Fully Managed": {"name": "Fully Managed", "included": true},
  "cta": "Contact Sales",
  "link": "https://innovz.surveysparrow.com/s/Zaakiy-onboarding/tt-NwNkd Contact",
  "DetailsAddress": "Heirs of Ahmed Obaid Bin Touq Al Marri Building, Al Mararr, Deira, Dubai, United Arab Emirates",
  "Phone number": "+971 52 867 8679"
}
"""


def test_contact_extraction():
    """Test contact information extraction"""
    print("=" * 60)
    print("Testing Contact Information Extraction")
    print("=" * 60)

    # Test extraction
    result = contact_extractor.extract_contact_info(TEST_CHUNK)

    print("\n📞 Extracted Contact Information:")
    print(f"  Phones: {result.get('phones', [])}")
    print(f"  Emails: {result.get('emails', [])}")
    print(f"  Addresses: {result.get('addresses', [])}")
    print(f"  Demo Links: {result.get('demo_links', [])}")
    print(f"  All Links: {result.get('links', [])}")
    print(f"  Has Contact Info: {result.get('has_contact_info', False)}")

    # Verify extraction
    assert len(result.get("phones", [])) > 0, "❌ Phone number not extracted!"
    assert len(result.get("demo_links", [])) > 0, "❌ Demo link not extracted!"
    assert len(result.get("addresses", [])) > 0, "❌ Address not extracted!"
    assert result.get("has_contact_info", False), "❌ Contact info flag not set!"

    print("\n✅ All contact information extracted successfully!")

    # Test scoring
    score = contact_extractor.score_chunk_for_contact_query(TEST_CHUNK)
    print(f"\n📊 Contact Score: {score:.2f}")
    print(f"  (Higher score = more likely to contain contact info)")

    return result


def test_phone_patterns():
    """Test various phone number formats"""
    print("\n" + "=" * 60)
    print("Testing Phone Number Patterns")
    print("=" * 60)

    test_phones = [
        "+971 52 867 8679",
        "+971528678679",
        "971 52 867 8679",
        "052 867 8679",
        "52 867 8679",
    ]

    for phone in test_phones:
        test_text = f"Contact us at {phone} for more information."
        result = contact_extractor.extract_contact_info(test_text)
        phones_found = result.get("phones", [])
        print(f"\n  Input: {phone}")
        print(f"  Found: {phones_found}")
        assert len(phones_found) > 0, f"❌ Phone {phone} not extracted!"


def test_demo_link_extraction():
    """Test demo/booking link extraction"""
    print("\n" + "=" * 60)
    print("Testing Demo Link Extraction")
    print("=" * 60)

    test_links = [
        "https://innovz.surveysparrow.com/s/Zaakiy-onboarding/tt-NwNkd",
        "https://calendly.com/demo",
        "https://example.com/book-demo",
        "www.example.com/schedule-appointment",
    ]

    for link in test_links:
        test_text = f"Book a demo at {link}"
        result = contact_extractor.extract_contact_info(test_text)
        demo_links = result.get("demo_links", [])
        print(f"\n  Input: {link}")
        print(f"  Found in demo_links: {demo_links}")
        if (
            "surveysparrow" in link.lower()
            or "calendly" in link.lower()
            or "demo" in link.lower()
            or "book" in link.lower()
        ):
            assert len(demo_links) > 0, f"❌ Demo link {link} not identified!"


if __name__ == "__main__":
    print("\n🧪 Running Contact Retrieval Tests\n")

    try:
        # Test 1: Basic extraction
        result = test_contact_extraction()

        # Test 2: Phone patterns
        test_phone_patterns()

        # Test 3: Demo links
        test_demo_link_extraction()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
