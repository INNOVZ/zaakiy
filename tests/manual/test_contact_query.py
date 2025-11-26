"""
Test script to verify contact query responses include actual contact information
"""

import asyncio
import os
import re
import sys

from dotenv import load_dotenv

# Add backend root to path
backend_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_root))
load_dotenv()

from app.services.chat.chat_service import ChatService


async def test_contact_query():
    """Test contact query to verify actual contact info is returned"""

    org_id = "2f97237c-9129-4a90-841f-2ffb7a632745"
    chatbot_id = "06acbb58-58b3-48a8-bdaa-57522f7b97e4"

    # Initialize chat service
    chatbot_config = {
        "id": chatbot_id,
        "org_id": org_id,
        "name": "Ohh Zone",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 500,
        "system_instructions": "You are a helpful AI assistant.",
    }

    chat_service = ChatService(org_id=org_id, chatbot_config=chatbot_config)

    print("=" * 100)
    print("🧪 TESTING: Contact Query Response")
    print("=" * 100)
    print()

    # Test queries
    test_queries = [
        "how can i contact you?",
        "what's your phone number?",
        "what is your email address?",
        "how can I reach you?",
    ]

    # Expected contact info (from Pinecone/website)
    expected_phones = [
        "+91 77 366 49 722",
        "+91 79 075 68380",
        "917736649722",
        "917907568380",
    ]
    expected_emails = [
        "hello@ohhzones.com",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"📝 TEST {i}/{len(test_queries)}: {query}")
        print("-" * 100)
        print()

        try:
            # Process message
            response = await chat_service.process_message(
                message=query,
                session_id=f"test-contact-{i}",
                end_user_identifier="test-user",
            )

            assistant_response = response.get("response", "")
            sources = response.get("sources", [])
            context_used = response.get("context_used", "")
            contact_info = response.get("contact_info", {})
            document_count = response.get("document_count", 0)

            print(f"📊 RESPONSE STATS:")
            print(f"   Response length: {len(assistant_response)} chars")
            print(f"   Documents retrieved: {document_count}")
            print(f"   Context length: {len(context_used)} chars")
            print(f"   Sources: {len(sources)}")
            print()

            # Check contact info in response metadata
            phones_in_metadata = contact_info.get("phones", [])
            emails_in_metadata = contact_info.get("emails", [])

            print(f"📞 CONTACT INFO IN METADATA:")
            print(f"   Phones: {phones_in_metadata}")
            print(f"   Emails: {emails_in_metadata}")
            print()

            # Extract contact info from response text
            phone_pattern = r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
            email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"

            phones_in_response = re.findall(phone_pattern, assistant_response)
            emails_in_response = re.findall(email_pattern, assistant_response)

            print(f"📞 CONTACT INFO IN RESPONSE TEXT:")
            print(f"   Phones found: {phones_in_response}")
            print(f"   Emails found: {emails_in_response}")
            print()

            # Verify against expected values
            found_expected_phone = False
            found_expected_email = False

            for phone in phones_in_response:
                # Normalize phone (remove spaces, dashes)
                normalized = re.sub(r"[^\d+]", "", phone)
                for expected in expected_phones:
                    expected_normalized = re.sub(r"[^\d+]", "", expected)
                    if (
                        normalized in expected_normalized
                        or expected_normalized in normalized
                    ):
                        found_expected_phone = True
                        break
                if found_expected_phone:
                    break

            for email in emails_in_response:
                if email.lower() in [e.lower() for e in expected_emails]:
                    found_expected_email = True
                    break

            # Results
            print(f"✅ VERIFICATION:")
            if found_expected_phone:
                print(f"   ✅ Phone number found in response!")
            else:
                print(f"   ❌ Phone number NOT found in response")
                print(f"      Expected: {expected_phones[0]}")
                print(f"      Found: {phones_in_response}")

            if found_expected_email:
                print(f"   ✅ Email address found in response!")
            else:
                print(f"   ❌ Email address NOT found in response")
                print(f"      Expected: {expected_emails[0]}")
                print(f"      Found: {emails_in_response}")

            print()

            # Check for vague responses
            vague_phrases = [
                "connect with our team",
                "contact us",
                "reach out",
                "get in touch",
            ]

            has_vague = any(
                phrase in assistant_response.lower() for phrase in vague_phrases
            )
            has_specific = found_expected_phone or found_expected_email

            if has_specific:
                print(f"✅ RESPONSE QUALITY: Specific contact info provided")
            elif has_vague and not has_specific:
                print(
                    f"⚠️  RESPONSE QUALITY: Vague response (no specific contact info)"
                )
            else:
                print(f"ℹ️  RESPONSE QUALITY: Neutral")

            print()

            # Show response preview
            print(f"💬 RESPONSE PREVIEW (first 300 chars):")
            print(f"   {assistant_response[:300]}...")
            print()

            # Show context preview (contact section)
            if "CONTACT INFORMATION" in context_used:
                contact_section_start = context_used.find("CONTACT INFORMATION")
                contact_section = context_used[
                    contact_section_start : contact_section_start + 500
                ]
                print(f"📋 CONTACT SECTION IN CONTEXT:")
                print(f"   {contact_section}")
                print()

            print("=" * 100)
            print()

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            print()
            print("=" * 100)
            print()


if __name__ == "__main__":
    asyncio.run(test_contact_query())
