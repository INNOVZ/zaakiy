"""WhatsApp message formatting utilities."""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class WhatsAppFormatter:
    """Format messages for WhatsApp platform."""

    # WhatsApp formatting syntax
    BOLD = "*{text}*"
    ITALIC = "_{text}_"
    STRIKETHROUGH = "~{text}~"
    MONOSPACE = "```{text}```"

    @staticmethod
    def format_message(
        text: str,
        max_length: int = 1600,
        preserve_formatting: bool = True,
    ) -> str:
        """
        Format message text for WhatsApp.

        Args:
            text: Message text to format
            max_length: Maximum message length (default 1600 for cost efficiency)
            preserve_formatting: Whether to convert markdown to WhatsApp formatting

        Returns:
            Formatted message text
        """
        # Convert markdown to WhatsApp formatting FIRST (before truncation)
        # This prevents breaking markdown tags during truncation
        if preserve_formatting:
            text = WhatsAppFormatter._markdown_to_whatsapp(text)

        # Truncate if needed (after formatting to preserve formatting integrity)
        if len(text) > max_length:
            truncated = text[: max_length - 3] + "..."
            logger.debug(
                "Message truncated from %d to %d characters", len(text), len(truncated)
            )
            text = truncated

        return text

    @staticmethod
    def _markdown_to_whatsapp(text: str) -> str:
        """Convert markdown formatting to WhatsApp formatting."""
        # Step 1: Convert bold first: **text** or __text__ -> *text*
        # Use a unique placeholder that won't be matched by italic regex
        # Using a pattern with special chars that won't match * or _
        bold_start = (
            "\uE000"  # Private Use Area character (won't appear in normal text)
        )
        bold_end = "\uE001"
        text = re.sub(r"\*\*(.+?)\*\*", rf"{bold_start}\1{bold_end}", text)
        text = re.sub(r"__(.+?)__", rf"{bold_start}\1{bold_end}", text)

        # Step 2: Convert italic: *text* or _text_ -> _text_
        # Handle markdown italic: *text* (single asterisk, not bold)
        # Match *text* where it's not part of **text** (bold) - already handled above
        text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"_\1_", text)

        # Handle markdown italic: _text_ (single underscore, not bold)
        # Match _text_ where underscores are not part of __text__ (bold) - already handled above
        text = re.sub(r"(?<!_)_([^_]+?)_(?!_)", r"_\1_", text)

        # Step 3: Replace bold placeholder with WhatsApp bold format: *text*
        text = text.replace(bold_start, "*").replace(bold_end, "*")

        # Code blocks: ```code``` -> ```code```
        # Already in WhatsApp format, just ensure proper formatting
        text = re.sub(r"```(.+?)```", r"```\1```", text, flags=re.DOTALL)

        # Inline code: `code` -> ```code```
        text = re.sub(r"`(.+?)`", r"```\1```", text)

        return text

    @staticmethod
    def split_long_message(text: str, max_length: int = 1600) -> List[str]:
        """
        Split long messages into multiple parts.

        WhatsApp supports up to 4096 characters, but splitting helps with
        readability and ensures delivery even with network issues.

        Args:
            text: Message text to split
            max_length: Maximum length per part

        Returns:
            List of message parts
        """
        if len(text) <= max_length:
            return [text]

        parts = []
        current_part = ""

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # If adding this paragraph would exceed limit, save current part
            if len(current_part) + len(paragraph) + 2 > max_length:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""

            # If single paragraph is too long, split by sentences
            if len(paragraph) > max_length:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""

                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    # Handle sentences that exceed max_length individually
                    if len(sentence) > max_length:
                        # If current_part has content, save it first
                        if current_part:
                            parts.append(current_part.strip())
                            current_part = ""
                        # Split the long sentence itself (fallback: hard truncate)
                        # Try to split at word boundaries within the sentence
                        words = sentence.split()
                        long_sentence_part = ""
                        for word in words:
                            if len(long_sentence_part) + len(word) + 1 > max_length:
                                if long_sentence_part:
                                    parts.append(long_sentence_part.strip())
                                    long_sentence_part = ""
                            # If single word exceeds max_length, truncate it
                            if len(word) > max_length:
                                if long_sentence_part:
                                    parts.append(long_sentence_part.strip())
                                    long_sentence_part = ""
                                # Hard truncate with ellipsis
                                truncated_word = word[: max_length - 3] + "..."
                                parts.append(truncated_word)
                                continue
                            long_sentence_part += word + " "
                        if long_sentence_part.strip():
                            current_part = long_sentence_part
                    elif len(current_part) + len(sentence) + 1 > max_length:
                        if current_part:
                            parts.append(current_part.strip())
                            current_part = ""
                        current_part += sentence + " "
                    else:
                        current_part += sentence + " "
            else:
                current_part += paragraph + "\n\n"

        # Add remaining part
        if current_part.strip():
            parts.append(current_part.strip())

        return parts

    @staticmethod
    def escape_special_characters(text: str) -> str:
        """
        Escape special WhatsApp formatting characters if needed.

        Note: This is optional - WhatsApp will interpret formatting by default.
        Use this if you want to send literal * _ ~ characters.
        """
        # For literal characters, we'd need to escape, but WhatsApp doesn't
        # have a standard escape mechanism. This is a placeholder.
        return text
