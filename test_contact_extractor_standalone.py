"""
Standalone test for contact extractor (no dependencies)
"""
import json
import re


class ContactExtractor:
    """Extract contact information from text chunks"""

    PHONE_PATTERNS = [
        r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        r"\+?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}",
        r"\+971[-.\s]?\d{1,2}[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\d{2,3}[-.\s]?\d{3}[-.\s]?\d{4}",
    ]

    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    URL_PATTERNS = [
        r"https?://[^\s<>\"{}|\\^`\[\]]{1,2000}",
        r"www\.[^\s<>\"{}|\\^`\[\]]{1,2000}",
    ]
    DEMO_KEYWORDS = [
        "demo",
        "booking",
        "book",
        "schedule",
        "appointment",
        "onboarding",
        "surveysparrow",
        "calendly",
        "zoom",
        "meet",
        "contact",
        "form",
    ]

    def extract_contact_info(self, chunk: str):
        result = {
            "phones": [],
            "emails": [],
            "addresses": [],
            "links": [],
            "demo_links": [],
            "has_contact_info": False,
        }

        if not chunk or not isinstance(chunk, str):
            return result

        parsed_chunk = self._try_parse_json(chunk)
        if parsed_chunk != chunk:
            chunk = self._extract_text_from_json(parsed_chunk)

        result["phones"] = self._extract_phones(chunk)
        result["emails"] = self._extract_emails(chunk)
        result["addresses"] = self._extract_addresses(chunk)
        result["links"] = self._extract_links(chunk)
        result["demo_links"] = self._identify_demo_links(result["links"])
        result["has_contact_info"] = bool(
            result["phones"]
            or result["emails"]
            or result["addresses"]
            or result["demo_links"]
        )
        return result

    def _try_parse_json(self, text: str):
        if not text or not isinstance(text, str):
            return text
        try:
            return json.loads(text)
        except:
            pass
        json_match = re.search(
            r'\{[^{}]*"(?:phone|email|address|link|demo|contact)[^{}]*\}',
            text,
            re.IGNORECASE,
        )
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        return text

    def _extract_text_from_json(self, parsed):
        if isinstance(parsed, str):
            return parsed
        elif isinstance(parsed, dict):
            texts = []
            for key, value in parsed.items():
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, (dict, list)):
                    texts.append(self._extract_text_from_json(value))
            return " ".join(texts)
        elif isinstance(parsed, list):
            texts = [self._extract_text_from_json(item) for item in parsed]
            return " ".join(texts)
        else:
            return str(parsed)

    def _extract_phones(self, text: str):
        phones = set()
        for pattern in self.PHONE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = re.sub(r"[\s\-\(\)\.]", "", match.strip())
                if len(re.sub(r"\D", "", cleaned)) >= 10:
                    phones.add(match.strip())
        return list(phones)

    def _extract_emails(self, text: str):
        emails = set(re.findall(self.EMAIL_PATTERN, text, re.IGNORECASE))
        noise_patterns = [r"example\.com", r"test\.com", r"noreply", r"no-reply"]
        filtered = []
        for email in emails:
            if not any(
                re.search(pattern, email, re.IGNORECASE) for pattern in noise_patterns
            ):
                filtered.append(email.lower())
        return filtered

    def _extract_addresses(self, text: str):
        addresses = []
        address_indicators = [
            r"Building[^.!?]{10,100}",
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Building|Street|Road|Avenue)[^.!?]{10,100}",
            r"[^.!?]{20,150}(?:Dubai|Abu Dhabi|Sharjah|UAE|United Arab Emirates)[^.!?]{0,50}",
        ]
        for pattern in address_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = re.sub(r"\s+", " ", match.strip())
                if len(cleaned) > 15 and cleaned not in addresses:
                    addresses.append(cleaned)
        return addresses

    def _extract_links(self, text: str):
        links = set()
        for pattern in self.URL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = re.sub(r"[.,;!?]+$", "", match.strip())
                if cleaned.startswith("www."):
                    cleaned = "https://" + cleaned
                if cleaned.startswith("http://") or cleaned.startswith("https://"):
                    links.add(cleaned)
        return list(links)

    def _identify_demo_links(self, links: list):
        demo_links = []
        for link in links:
            link_lower = link.lower()
            if any(keyword in link_lower for keyword in self.DEMO_KEYWORDS):
                demo_links.append(link)
            elif any(
                platform in link_lower
                for platform in [
                    "calendly",
                    "surveysparrow",
                    "cal.com",
                    "acuity",
                    "appointlet",
                    "schedule",
                    "booking",
                ]
            ):
                demo_links.append(link)
        return demo_links


# Test data
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


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Contact Information Extraction")
    print("=" * 60)

    extractor = ContactExtractor()
    result = extractor.extract_contact_info(TEST_CHUNK)

    print("\n📞 Extracted Contact Information:")
    print(f"  Phones: {result.get('phones', [])}")
    print(f"  Emails: {result.get('emails', [])}")
    print(f"  Addresses: {result.get('addresses', [])}")
    print(f"  Demo Links: {result.get('demo_links', [])}")
    print(f"  All Links: {result.get('links', [])}")
    print(f"  Has Contact Info: {result.get('has_contact_info', False)}")

    # Verify
    assert len(result.get("phones", [])) > 0, "❌ Phone number not extracted!"
    assert len(result.get("demo_links", [])) > 0, "❌ Demo link not extracted!"
    assert len(result.get("addresses", [])) > 0, "❌ Address not extracted!"

    print("\n✅ All contact information extracted successfully!")
    print("=" * 60)
