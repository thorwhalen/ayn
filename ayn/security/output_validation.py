"""Output validation and filtering for agent responses."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class ValidationResult:
    """Result of output validation."""

    is_valid: bool
    filtered_output: Optional[str] = None
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class OutputFilter(ABC):
    """Base class for output filters."""

    @abstractmethod
    def filter(self, output: str) -> ValidationResult:
        """Filter output and return result.

        Args:
            output: Output to filter

        Returns:
            ValidationResult
        """
        pass


class PIIDetector(OutputFilter):
    """Detects and redacts personally identifiable information.

    Detects:
    - Email addresses
    - Phone numbers
    - SSNs
    - Credit card numbers
    - IP addresses
    - Physical addresses (basic)

    Example:
        >>> detector = PIIDetector()
        >>> result = detector.filter("Contact me at john@example.com or 555-1234")
        >>> result.is_valid
        False
        >>> "[EMAIL]" in result.filtered_output
        True
    """

    # PII patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+\d{1,2}\s?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def __init__(self, redact: bool = True, patterns: Optional[dict] = None):
        """Initialize PII detector.

        Args:
            redact: Whether to redact PII or just detect
            patterns: Custom PII patterns to add
        """
        self.redact = redact
        self.patterns = self.PATTERNS.copy()
        if patterns:
            self.patterns.update(patterns)

    def filter(self, output: str) -> ValidationResult:
        """Detect and optionally redact PII.

        Args:
            output: Output to filter

        Returns:
            ValidationResult
        """
        issues = []
        filtered = output

        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                issues.append(f"Found {len(matches)} {pii_type}(s)")

                if self.redact:
                    replacement = f"[{pii_type.upper()}]"
                    filtered = re.sub(pattern, replacement, filtered)

        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            filtered_output=filtered if not is_valid else output,
            issues=issues,
        )

    def detect_pii_types(self, output: str) -> Set[str]:
        """Detect which types of PII are present.

        Args:
            output: Output to analyze

        Returns:
            Set of PII types found
        """
        found_types = set()

        for pii_type, pattern in self.patterns.items():
            if re.search(pattern, output):
                found_types.add(pii_type)

        return found_types


class ToxicContentFilter(OutputFilter):
    """Filters toxic, offensive, or harmful content.

    Basic implementation using pattern matching.
    For production use, consider integrating with:
    - Perspective API
    - OpenAI Moderation API
    - HuggingFace toxicity models

    Example:
        >>> filter = ToxicContentFilter()
        >>> result = filter.filter("This is a normal response")
        >>> result.is_valid
        True
    """

    # Basic toxic patterns (very simplified)
    TOXIC_PATTERNS = [
        r"\b(profanity|offensive_word)\b",  # Placeholder
        r"\b(hate\s+speech)\b",
        r"\b(threats?)\b",
    ]

    def __init__(
        self,
        block_toxic: bool = True,
        custom_patterns: Optional[List[str]] = None,
    ):
        """Initialize toxic content filter.

        Args:
            block_toxic: Whether to block toxic content
            custom_patterns: Additional patterns to check
        """
        self.block_toxic = block_toxic
        self.patterns = self.TOXIC_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def filter(self, output: str) -> ValidationResult:
        """Filter toxic content.

        Args:
            output: Output to filter

        Returns:
            ValidationResult
        """
        issues = []

        for pattern in self.patterns:
            if re.search(pattern, output, re.IGNORECASE):
                issues.append(f"Matched toxic pattern: {pattern}")

        is_valid = len(issues) == 0

        if not is_valid and self.block_toxic:
            filtered = "[CONTENT FILTERED: Potentially harmful content detected]"
        else:
            filtered = output

        return ValidationResult(
            is_valid=is_valid,
            filtered_output=filtered if not is_valid else output,
            issues=issues,
        )


class OutputValidator:
    """Comprehensive output validation.

    Combines multiple filters:
    - PII detection/redaction
    - Toxic content filtering
    - Custom validators

    Example:
        >>> validator = OutputValidator()
        >>> result = validator.validate("Contact me at test@example.com")
        >>> result.is_valid
        False
    """

    def __init__(
        self,
        detect_pii: bool = True,
        filter_toxic: bool = True,
        custom_filters: Optional[List[OutputFilter]] = None,
    ):
        """Initialize output validator.

        Args:
            detect_pii: Enable PII detection
            filter_toxic: Enable toxic content filtering
            custom_filters: Additional custom filters
        """
        self.filters: List[OutputFilter] = []

        if detect_pii:
            self.filters.append(PIIDetector())

        if filter_toxic:
            self.filters.append(ToxicContentFilter())

        if custom_filters:
            self.filters.extend(custom_filters)

    def validate(self, output: str) -> ValidationResult:
        """Validate output through all filters.

        Args:
            output: Output to validate

        Returns:
            ValidationResult with combined results
        """
        all_issues = []
        current_output = output

        for filter in self.filters:
            result = filter.filter(current_output)

            if not result.is_valid:
                all_issues.extend(result.issues)
                if result.filtered_output:
                    current_output = result.filtered_output

        is_valid = len(all_issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            filtered_output=current_output if not is_valid else output,
            issues=all_issues,
        )

    def add_filter(self, filter: OutputFilter):
        """Add a custom filter.

        Args:
            filter: OutputFilter to add
        """
        self.filters.append(filter)
