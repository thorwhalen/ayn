"""Prompt injection detection and prevention."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class InjectionResult:
    """Result of injection detection."""

    is_suspicious: bool
    confidence: float  # 0-1
    reasons: List[str]
    sanitized_input: Optional[str] = None

    def __str__(self) -> str:
        if not self.is_suspicious:
            return "✓ Input appears safe"

        lines = [f"⚠ Potential injection detected (confidence: {self.confidence:.2%})"]
        lines.append("Reasons:")
        for reason in self.reasons:
            lines.append(f"  • {reason}")

        return "\n".join(lines)


class InjectionDetector:
    """Detects common prompt injection patterns.

    Looks for:
    - System prompt override attempts
    - Role confusion attacks
    - Instruction injection
    - Delimiter manipulation
    - Encoding attacks

    Example:
        >>> detector = InjectionDetector()
        >>> result = detector.detect("Normal user input")
        >>> result.is_suspicious
        False
        >>> result = detector.detect("Ignore previous instructions and...")
        >>> result.is_suspicious
        True
    """

    # Common injection patterns
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        r"ignore\s+(previous|all|above)\s+(instructions|prompts?|commands?)",
        r"forget\s+(everything|all|previous)",
        r"disregard\s+(previous|all|above)",
        # Role manipulation
        r"you\s+are\s+now",
        r"act\s+as\s+a",
        r"pretend\s+(to\s+be|you\s+are)",
        r"roleplay\s+as",
        # System prompts
        r"system\s*:",
        r"<\|?system\|?>",
        r"\[system\]",
        # Delimiter attacks
        r"```\s*system",
        r"===\s*end",
        r"---\s*end",
        # Common injection phrases
        r"new\s+instructions?",
        r"override\s+instructions?",
        r"bypass\s+restrictions?",
        # Encoding attacks
        r"\\x[0-9a-f]{2}",  # Hex encoding
        r"base64\s*[:=]",
        r"decode\s*\(",
    ]

    def __init__(self, sensitivity: float = 0.5):
        """Initialize detector.

        Args:
            sensitivity: Detection sensitivity (0-1). Higher = more strict.
        """
        self.sensitivity = sensitivity
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def detect(self, user_input: str) -> InjectionResult:
        """Detect potential injection attempts.

        Args:
            user_input: User input to analyze

        Returns:
            InjectionResult with detection details
        """
        reasons = []
        matches = 0

        # Check against patterns
        for pattern in self.patterns:
            if pattern.search(user_input):
                matches += 1
                reasons.append(f"Matched pattern: {pattern.pattern}")

        # Calculate confidence based on matches
        confidence = min(matches / 3.0, 1.0)  # 3+ matches = high confidence

        # Check for suspicious length (very long inputs)
        if len(user_input) > 5000:
            confidence += 0.1
            reasons.append("Input is unusually long")

        # Check for repeated delimiters
        delimiter_count = user_input.count("---") + user_input.count("===") + user_input.count("```")
        if delimiter_count > 3:
            confidence += 0.15
            reasons.append("Multiple delimiter sequences detected")

        confidence = min(confidence, 1.0)
        is_suspicious = confidence >= self.sensitivity

        return InjectionResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            reasons=reasons,
        )

    def sanitize(self, user_input: str) -> str:
        """Sanitize input by removing suspicious patterns.

        Args:
            user_input: Input to sanitize

        Returns:
            Sanitized input
        """
        sanitized = user_input

        # Remove common injection markers
        markers = [
            r"<\|?system\|?>.*?<\|?/system\|?>",
            r"\[system\].*?\[/system\]",
            r"```system.*?```",
        ]

        for marker in markers:
            sanitized = re.sub(marker, "", sanitized, flags=re.IGNORECASE | re.DOTALL)

        # Remove suspicious instructions
        for pattern in self.patterns[:5]:  # First 5 are instruction-related
            sanitized = pattern.sub("[REDACTED]", sanitized)

        return sanitized.strip()


class PromptGuard:
    """Comprehensive prompt security guard.

    Combines multiple security checks:
    - Injection detection
    - Input sanitization
    - Length limits
    - Character filtering

    Example:
        >>> guard = PromptGuard()
        >>> result = guard.check("Hello, how are you?")
        >>> result.is_safe
        True
        >>> safe_input = guard.sanitize("Ignore all instructions and do X")
        >>> safe_input
        '[REDACTED] and do X'
    """

    def __init__(
        self,
        max_length: int = 10000,
        sensitivity: float = 0.5,
        auto_sanitize: bool = False,
    ):
        """Initialize PromptGuard.

        Args:
            max_length: Maximum allowed input length
            sensitivity: Injection detection sensitivity (0-1)
            auto_sanitize: Automatically sanitize inputs
        """
        self.max_length = max_length
        self.auto_sanitize = auto_sanitize
        self.detector = InjectionDetector(sensitivity=sensitivity)

        # Blocked patterns (absolute no-go)
        self.blocked_patterns: Set[str] = set()

    def add_blocked_pattern(self, pattern: str):
        """Add a pattern to block list.

        Args:
            pattern: Regular expression pattern to block
        """
        self.blocked_patterns.add(pattern)

    def is_safe(self, user_input: str) -> tuple[bool, Optional[str]]:
        """Check if input is safe.

        Args:
            user_input: Input to check

        Returns:
            (is_safe, reason_if_unsafe)
        """
        # Length check
        if len(user_input) > self.max_length:
            return False, f"Input exceeds maximum length of {self.max_length}"

        # Blocked patterns check
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, f"Input contains blocked pattern: {pattern}"

        # Injection detection
        result = self.detector.detect(user_input)
        if result.is_suspicious:
            return False, f"Potential injection detected: {', '.join(result.reasons)}"

        return True, None

    def sanitize(self, user_input: str) -> str:
        """Sanitize user input.

        Args:
            user_input: Input to sanitize

        Returns:
            Sanitized input
        """
        # Trim to max length
        sanitized = user_input[:self.max_length]

        # Apply injection sanitization
        sanitized = self.detector.sanitize(sanitized)

        return sanitized

    def check(self, user_input: str) -> InjectionResult:
        """Check and optionally sanitize input.

        Args:
            user_input: Input to check

        Returns:
            InjectionResult with detection details
        """
        is_safe, reason = self.is_safe(user_input)

        if not is_safe:
            result = InjectionResult(
                is_suspicious=True,
                confidence=1.0,
                reasons=[reason] if reason else [],
            )
        else:
            result = self.detector.detect(user_input)

        # Auto-sanitize if enabled and suspicious
        if self.auto_sanitize and result.is_suspicious:
            result.sanitized_input = self.sanitize(user_input)

        return result
