"""
WhatsApp Response Formatter
----------------------------
Formats AI responses specifically for WhatsApp with natural, human-like structure.
No bold formatting - just clean, conversational responses.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WhatsAppResponseFormatter:
    """
    Formats AI responses for optimal display in WhatsApp.

    Features:
    - Natural, conversational tone
    - Clean structure with emojis
    - Clickable links
    - Contact information formatting
    - No bold asterisks - human-like responses
    """

    # Maximum message length for WhatsApp
    MAX_LENGTH = 4096

    @classmethod
    def format_for_whatsapp(
        cls,
        response_text: str,
        context_data: Optional[Dict[str, Any]] = None,
        intent_type: Optional[str] = None,
    ) -> str:
        """
        Format response for WhatsApp with natural, human-like structure.

        Args:
            response_text: Raw AI response
            context_data: Context information (contact, products, etc.)
            intent_type: Type of query (contact, pricing, product, etc.)

        Returns:
            Formatted WhatsApp message
        """
        if not response_text:
            return "I'm here to help! How can I assist you today? 😊"

        # Apply formatting based on intent
        if intent_type == "contact":
            formatted = cls._format_contact_response(response_text, context_data)
        elif intent_type == "pricing":
            formatted = cls._format_pricing_response(response_text, context_data)
        elif intent_type == "product":
            formatted = cls._format_product_response(response_text, context_data)
        elif intent_type == "booking":
            formatted = cls._format_booking_response(response_text, context_data)
        else:
            formatted = cls._format_general_response(response_text, context_data)

        # Apply general formatting improvements
        formatted = cls._enhance_formatting(formatted)

        # Ensure proper length
        formatted = cls._ensure_length(formatted)

        return formatted

    @classmethod
    def _format_contact_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format contact information responses naturally"""

        if not context_data:
            return response

        contact_info = context_data.get("contact_info", {})

        # Build natural contact info
        parts = []

        # Add the main response first
        if response and len(response) > 20:
            parts.append(response)
            parts.append("")

        # Phone numbers
        phones = contact_info.get("phones", [])
        if phones:
            if len(phones) == 1:
                parts.append(f"📱 {phones[0]}")
            else:
                parts.append("You can reach us at:")
                for phone in phones[:3]:
                    parts.append(f"  📱 {phone}")
            parts.append("")

        # Emails
        emails = contact_info.get("emails", [])
        if emails:
            if len(emails) == 1:
                parts.append(f"✉️ {emails[0]}")
            else:
                parts.append("Email us at:")
                for email in emails[:3]:
                    parts.append(f"  ✉️ {email}")
            parts.append("")

        # Addresses
        addresses = contact_info.get("addresses", [])
        if addresses:
            parts.append("📍 " + addresses[0])
            parts.append("")

        # Demo/Booking links
        demo_links = contact_info.get("demo_links", [])
        if demo_links:
            parts.append("Book a time that works for you:")
            parts.append(f"🔗 {demo_links[0]}")

        return "\n".join(parts)

    @classmethod
    def _format_pricing_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format pricing information naturally"""

        # Just clean up the response and add structure
        parts = []

        # Add emoji header if not present
        if "💰" not in response[:20] and "pricing" in response.lower()[:50]:
            parts.append("💰 Here's our pricing:")
            parts.append("")

        # Add the response
        parts.append(response)

        # Add CTA if demo links available
        if context_data:
            demo_links = context_data.get("contact_info", {}).get("demo_links", [])
            if demo_links and "demo" not in response.lower():
                parts.append("")
                parts.append("Want to see it in action? Book a demo:")
                parts.append(f"🔗 {demo_links[0]}")

        return "\n".join(parts)

    @classmethod
    def _format_product_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format product information naturally"""

        parts = []

        # Add emoji if not present
        if "🛍️" not in response[:20] and any(
            word in response.lower()[:50] for word in ["product", "item", "catalog"]
        ):
            parts.append("🛍️ " + response)
        else:
            parts.append(response)

        # Add product links if available
        if context_data:
            product_links = context_data.get("product_links", [])
            if product_links:
                parts.append("")
                parts.append("Check them out:")
                for product in product_links[:3]:
                    name = product.get("name", "Product")
                    url = product.get("url")
                    price = product.get("price")

                    product_line = f"  • {name}"
                    if price:
                        product_line += f" - {price}"
                    if url:
                        product_line += f"\n    🔗 {url}"

                    parts.append(product_line)

        return "\n".join(parts)

    @classmethod
    def _format_booking_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format booking/demo responses naturally"""

        parts = []

        # Add emoji if not present
        if "📅" not in response[:20]:
            parts.append("📅 " + response)
        else:
            parts.append(response)

        # Add booking link if available and not in response
        if context_data:
            demo_links = context_data.get("contact_info", {}).get("demo_links", [])
            if demo_links:
                # Check if link is already in response
                if demo_links[0] not in response:
                    parts.append("")
                    parts.append("Book your slot here:")
                    parts.append(f"🔗 {demo_links[0]}")

        return "\n".join(parts)

    @classmethod
    def _format_general_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format general responses naturally"""

        # Add greeting emoji if it's a greeting
        if any(
            greeting in response.lower()[:50]
            for greeting in ["hello", "hi", "hey", "welcome"]
        ):
            if not any(emoji in response[:10] for emoji in ["👋", "😊", "🙂"]):
                response = "👋 " + response

        # Structure bullet points naturally
        response = cls._structure_bullet_points(response)

        return response

    @classmethod
    def _structure_bullet_points(cls, text: str) -> str:
        """Convert plain lists to clean bullet points"""

        lines = text.split("\n")
        formatted_lines = []

        for line in lines:
            stripped = line.strip()

            # Convert various list markers to clean bullets
            if stripped.startswith(("- ", "* ", "• ")):
                # Remove old marker and add clean bullet
                content = re.sub(r"^[-*•]\s*", "", stripped)
                formatted_lines.append(f"  • {content}")
            elif re.match(r"^\d+[\.)]\s", stripped):
                # Keep numbered lists as is
                formatted_lines.append(f"  {stripped}")
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    @classmethod
    def _enhance_formatting(cls, text: str) -> str:
        """Apply general formatting enhancements"""

        # Remove any bold asterisks that might have been added by AI
        text = cls._remove_bold_formatting(text)

        # Ensure proper spacing around sections
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines

        # Format URLs to be more visible
        text = cls._format_urls(text)

        # Format phone numbers with emoji
        text = cls._format_phone_numbers(text)

        # Format emails with emoji
        text = cls._format_emails(text)

        return text

    @classmethod
    def _remove_bold_formatting(cls, text: str) -> str:
        """Remove bold asterisks for more natural text"""

        # Remove bold formatting: *text* -> text
        # But keep single asterisks that are not formatting
        text = re.sub(r"\*([^*]+)\*", r"\1", text)

        return text

    @classmethod
    def _format_urls(cls, text: str) -> str:
        """Ensure URLs are properly formatted"""

        # WhatsApp auto-detects URLs, just add emoji if not present
        url_pattern = r"(?<![🔗\s])(https?://[^\s]+)"

        def format_url(match):
            url = match.group(1)
            # Remove trailing punctuation
            url = re.sub(r"[.,;:!?]+$", "", url)
            # Check if emoji is already before URL
            start_pos = match.start()
            if start_pos > 2:
                before = text[max(0, start_pos - 3) : start_pos]
                if "🔗" in before:
                    return url
            return f"🔗 {url}"

        return re.sub(url_pattern, format_url, text)

    @classmethod
    def _format_phone_numbers(cls, text: str) -> str:
        """Add emoji to phone numbers if not present"""

        # Pattern for phone numbers
        phone_pattern = (
            r"(?<![📱\s])(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})"
        )

        def format_phone(match):
            phone = match.group(1)
            # Check if emoji is already before phone
            start_pos = match.start()
            if start_pos > 2:
                before = text[max(0, start_pos - 3) : start_pos]
                if "📱" in before:
                    return phone
            return f"📱 {phone}"

        return re.sub(phone_pattern, format_phone, text)

    @classmethod
    def _format_emails(cls, text: str) -> str:
        """Add emoji to email addresses if not present"""

        email_pattern = (
            r"(?<![✉️\s])\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
        )

        def format_email(match):
            email = match.group(1)
            # Check if emoji is already before email
            start_pos = match.start()
            if start_pos > 2:
                before = text[max(0, start_pos - 3) : start_pos]
                if "✉️" in before:
                    return email
            return f"✉️ {email}"

        return re.sub(email_pattern, format_email, text)

    @classmethod
    def _ensure_length(cls, text: str) -> str:
        """Ensure message fits WhatsApp length limits"""

        if len(text) <= cls.MAX_LENGTH:
            return text

        # Truncate and add continuation message
        truncated = text[: cls.MAX_LENGTH - 100]

        # Try to cut at a sentence boundary
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")

        cut_point = max(last_period, last_newline)
        if cut_point > cls.MAX_LENGTH - 200:
            truncated = truncated[: cut_point + 1]

        truncated += "\n\n(Message was too long - feel free to ask for more details!) 📝"

        return truncated

    @classmethod
    def create_welcome_message(cls, business_name: Optional[str] = None) -> str:
        """Create a welcoming first message"""

        name = business_name or "our team"

        return f"""👋 Hey there!

I'm the AI Assistant for {name}. I can help you with:

  • Product information 🛍️
  • Pricing details 💰
  • Booking demos 📅
  • Contact information 📞
  • Any questions you have ❓

What would you like to know? 😊"""

    @classmethod
    def create_error_message(cls) -> str:
        """Create a friendly error message"""

        return """😅 Oops, something went wrong on my end!

Could you try:
  • Rephrasing your question
  • Being a bit more specific
  • Asking one thing at a time

I'm here to help! 💪"""


# Convenience function
def format_whatsapp_response(
    response_text: str,
    context_data: Optional[Dict[str, Any]] = None,
    intent_type: Optional[str] = None,
) -> str:
    """
    Quick function to format response for WhatsApp.

    Args:
        response_text: Raw AI response
        context_data: Context information
        intent_type: Type of query

    Returns:
        Formatted WhatsApp message
    """
    return WhatsAppResponseFormatter.format_for_whatsapp(
        response_text, context_data, intent_type
    )


# Export
__all__ = ["WhatsAppResponseFormatter", "format_whatsapp_response"]


class WhatsAppResponseFormatter:
    """
    Formats AI responses for optimal display in WhatsApp.

    Features:
    - Proper emoji usage
    - Clean bullet points
    - Clickable links
    - Card-like structures
    - Contact information formatting
    - Pricing tables
    """

    # WhatsApp formatting characters
    BOLD = "*"
    ITALIC = "_"
    STRIKETHROUGH = "~"
    MONOSPACE = "```"

    # Maximum message length for WhatsApp
    MAX_LENGTH = 4096

    @classmethod
    def format_for_whatsapp(
        cls,
        response_text: str,
        context_data: Optional[Dict[str, Any]] = None,
        intent_type: Optional[str] = None,
    ) -> str:
        """
        Format response for WhatsApp with proper structure.

        Args:
            response_text: Raw AI response
            context_data: Context information (contact, products, etc.)
            intent_type: Type of query (contact, pricing, product, etc.)

        Returns:
            Formatted WhatsApp message
        """
        if not response_text:
            return "I'm here to help! How can I assist you today? 😊"

        # Apply formatting based on intent
        if intent_type == "contact":
            formatted = cls._format_contact_response(response_text, context_data)
        elif intent_type == "pricing":
            formatted = cls._format_pricing_response(response_text, context_data)
        elif intent_type == "product":
            formatted = cls._format_product_response(response_text, context_data)
        elif intent_type == "booking":
            formatted = cls._format_booking_response(response_text, context_data)
        else:
            formatted = cls._format_general_response(response_text, context_data)

        # Apply general formatting improvements
        formatted = cls._enhance_formatting(formatted)

        # Ensure proper length
        formatted = cls._ensure_length(formatted)

        return formatted

    @classmethod
    def _format_contact_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format contact information responses"""

        if not context_data:
            return response

        contact_info = context_data.get("contact_info", {})

        # Build structured contact card
        parts = []

        # Header
        parts.append("📞 *Contact Information*")
        parts.append("")

        # Phone numbers
        phones = contact_info.get("phones", [])
        if phones:
            parts.append("*Phone:*")
            for phone in phones[:3]:  # Limit to 3
                parts.append(f"  📱 {phone}")
            parts.append("")

        # Emails
        emails = contact_info.get("emails", [])
        if emails:
            parts.append("*Email:*")
            for email in emails[:3]:
                parts.append(f"  ✉️ {email}")
            parts.append("")

        # Addresses
        addresses = contact_info.get("addresses", [])
        if addresses:
            parts.append("*Address:*")
            for addr in addresses[:2]:
                parts.append(f"  📍 {addr}")
            parts.append("")

        # Demo/Booking links
        demo_links = contact_info.get("demo_links", [])
        if demo_links:
            parts.append("*Book a Demo:*")
            parts.append(f"  🔗 {demo_links[0]}")
            parts.append("")

        # Add original response if it has additional context
        if response and len(response) > 50:
            parts.append("ℹ️ " + response)

        return "\n".join(parts)

    @classmethod
    def _format_pricing_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format pricing information responses"""

        # Extract pricing from response
        pricing_pattern = r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:/month|/year|/mo|/yr)?"
        prices = re.findall(pricing_pattern, response)

        if not prices:
            return response

        # Structure the response
        parts = []
        parts.append("💰 *Pricing Plans*")
        parts.append("")

        # Try to extract plan names and prices
        lines = response.split("\n")
        current_plan = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line contains a price
            if re.search(pricing_pattern, line):
                # Format as a plan
                # Bold the plan name, keep price visible
                formatted_line = re.sub(r"([^$]+)(\$[^$]+)", r"*\1* \2", line)
                parts.append(f"  • {formatted_line}")
            elif line.startswith("-") or line.startswith("•"):
                # Feature list
                parts.append(f"    {line}")
            else:
                # Regular text
                parts.append(line)

        # Add CTA
        if context_data:
            demo_links = context_data.get("contact_info", {}).get("demo_links", [])
            if demo_links:
                parts.append("")
                parts.append("📅 *Ready to get started?*")
                parts.append(f"Book a demo: {demo_links[0]}")

        return "\n".join(parts)

    @classmethod
    def _format_product_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format product information responses"""

        if not context_data:
            return response

        product_links = context_data.get("product_links", [])

        if not product_links:
            return response

        # Build product catalog
        parts = []
        parts.append("🛍️ *Our Products*")
        parts.append("")

        # Add main response
        parts.append(response)
        parts.append("")

        # Add product cards
        for product in product_links[:5]:  # Limit to 5 products
            name = product.get("name", "Product")
            url = product.get("url")
            price = product.get("price")

            product_line = f"• *{name}*"
            if price:
                product_line += f" - {price}"
            if url:
                product_line += f"\n  🔗 {url}"

            parts.append(product_line)
            parts.append("")

        return "\n".join(parts)

    @classmethod
    def _format_booking_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format booking/demo responses"""

        parts = []
        parts.append("📅 *Schedule a Demo*")
        parts.append("")
        parts.append(response)
        parts.append("")

        if context_data:
            demo_links = context_data.get("contact_info", {}).get("demo_links", [])
            if demo_links:
                parts.append("*Quick Booking Link:*")
                parts.append(f"🔗 {demo_links[0]}")
                parts.append("")
                parts.append(
                    "_Click the link above to choose a time that works for you!_ ⏰"
                )

        return "\n".join(parts)

    @classmethod
    def _format_general_response(
        cls, response: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format general responses with structure"""

        # Add emoji to greeting if it's a greeting
        if any(
            greeting in response.lower()[:50]
            for greeting in ["hello", "hi", "hey", "welcome"]
        ):
            if not any(emoji in response[:10] for emoji in ["👋", "😊", "🙂"]):
                response = "👋 " + response

        # Structure bullet points
        response = cls._structure_bullet_points(response)

        # Add section headers
        response = cls._add_section_headers(response)

        return response

    @classmethod
    def _structure_bullet_points(cls, text: str) -> str:
        """Convert plain lists to properly formatted bullet points"""

        lines = text.split("\n")
        formatted_lines = []

        for line in lines:
            stripped = line.strip()

            # Convert various list markers to clean bullets
            if stripped.startswith(("- ", "* ", "• ")):
                # Remove old marker and add clean bullet
                content = re.sub(r"^[-*•]\s*", "", stripped)
                formatted_lines.append(f"  • {content}")
            elif re.match(r"^\d+[\.)]\s", stripped):
                # Numbered list
                content = re.sub(r"^\d+[\.)]\s*", "", stripped)
                formatted_lines.append(f"  {stripped}")
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    @classmethod
    def _add_section_headers(cls, text: str) -> str:
        """Add visual separation for sections"""

        # Detect section headers (lines ending with colon)
        lines = text.split("\n")
        formatted_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # If line ends with colon and next line is a bullet, make it bold
            if stripped.endswith(":") and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith(("•", "-", "*", "1", "2", "3")):
                    # Make header bold
                    formatted_lines.append(f"*{stripped}*")
                    continue

            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    @classmethod
    def _enhance_formatting(cls, text: str) -> str:
        """Apply general formatting enhancements"""

        # Ensure proper spacing around sections
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines

        # Format URLs to be clickable
        text = cls._format_urls(text)

        # Format phone numbers with emoji
        text = cls._format_phone_numbers(text)

        # Format emails with emoji
        text = cls._format_emails(text)

        return text

    @classmethod
    def _format_urls(cls, text: str) -> str:
        """Ensure URLs are properly formatted"""

        # WhatsApp auto-detects URLs, but we can make them more visible
        url_pattern = r"(https?://[^\s]+)"

        def format_url(match):
            url = match.group(1)
            # Remove trailing punctuation
            url = re.sub(r"[.,;:!?]+$", "", url)
            return f"🔗 {url}"

        return re.sub(url_pattern, format_url, text)

    @classmethod
    def _format_phone_numbers(cls, text: str) -> str:
        """Add emoji to phone numbers"""

        # Pattern for phone numbers
        phone_pattern = r"(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})"

        def format_phone(match):
            phone = match.group(1)
            # Only add emoji if not already present
            if "📱" not in text[max(0, match.start() - 5) : match.start()]:
                return f"📱 {phone}"
            return phone

        return re.sub(phone_pattern, format_phone, text)

    @classmethod
    def _format_emails(cls, text: str) -> str:
        """Add emoji to email addresses"""

        email_pattern = r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"

        def format_email(match):
            email = match.group(1)
            # Only add emoji if not already present
            if "✉️" not in text[max(0, match.start() - 5) : match.start()]:
                return f"✉️ {email}"
            return email

        return re.sub(email_pattern, format_email, text)

    @classmethod
    def _ensure_length(cls, text: str) -> str:
        """Ensure message fits WhatsApp length limits"""

        if len(text) <= cls.MAX_LENGTH:
            return text

        # Truncate and add continuation message
        truncated = text[: cls.MAX_LENGTH - 100]

        # Try to cut at a sentence boundary
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")

        cut_point = max(last_period, last_newline)
        if cut_point > cls.MAX_LENGTH - 200:
            truncated = truncated[: cut_point + 1]

        truncated += "\n\n_Message truncated. Please ask for more details if needed!_ 📝"

        return truncated

    @classmethod
    def create_welcome_message(cls, business_name: Optional[str] = None) -> str:
        """Create a welcoming first message"""

        name = business_name or "our team"

        return f"""👋 *Welcome!*

I'm the AI Assistant for {name}. I'm here to help you with:

  • Product information 🛍️
  • Pricing details 💰
  • Booking demos 📅
  • Contact information 📞
  • General questions ❓

How can I assist you today? 😊"""

    @classmethod
    def create_error_message(cls) -> str:
        """Create a friendly error message"""

        return """😅 *Oops!*

I encountered a small hiccup processing your request.

Could you please try:
  • Rephrasing your question
  • Being more specific
  • Asking one thing at a time

I'm here to help! 💪"""


# Convenience function
def format_whatsapp_response(
    response_text: str,
    context_data: Optional[Dict[str, Any]] = None,
    intent_type: Optional[str] = None,
) -> str:
    """
    Quick function to format response for WhatsApp.

    Args:
        response_text: Raw AI response
        context_data: Context information
        intent_type: Type of query

    Returns:
        Formatted WhatsApp message
    """
    return WhatsAppResponseFormatter.format_for_whatsapp(
        response_text, context_data, intent_type
    )


# Export
__all__ = ["WhatsAppResponseFormatter", "format_whatsapp_response"]
