"""Test runner for agent contract tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_name: str
    passed: bool
    latency_ms: float
    error: Optional[str] = None
    output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [f"{status}: {self.test_name} ({self.latency_ms:.2f}ms)"]

        if self.error:
            lines.append(f"  Error: {self.error}")

        return "\n".join(lines)


@dataclass
class TestSuite:
    """Collection of test cases for an agent.

    Example:
        >>> suite = TestSuite(name="MyAgent Tests")
        >>> suite.add_test("test_basic", {"input": "hello"}, expected_output="HELLO")
        >>> # Run with controller
    """

    name: str
    tests: List[Dict[str, Any]] = field(default_factory=list)

    def add_test(
        self,
        name: str,
        input_data: Any,
        expected_output: Optional[Any] = None,
        validator: Optional[Callable[[Any], bool]] = None,
    ):
        """Add a test case to the suite.

        Args:
            name: Test name
            input_data: Input to pass to agent
            expected_output: Expected output (if None, just checks for no errors)
            validator: Custom validation function
        """
        self.tests.append(
            {
                "name": name,
                "input": input_data,
                "expected_output": expected_output,
                "validator": validator,
            }
        )

    def add_tests_from_dict(self, tests: Dict[str, Dict[str, Any]]):
        """Add multiple tests from a dictionary.

        Args:
            tests: Dict mapping test names to test configs
        """
        for name, config in tests.items():
            self.add_test(
                name=name,
                input_data=config["input"],
                expected_output=config.get("expected_output"),
                validator=config.get("validator"),
            )


class TestRunner:
    """Runs test suites against agent controllers.

    Example:
        >>> from ayn import TestRunner, TestSuite
        >>> suite = TestSuite(name="Test Agent")
        >>> suite.add_test("test1", {"text": "hello"}, expected_output={"result": "HELLO"})
        >>> runner = TestRunner()
        >>> # results = runner.run(controller, suite)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(
        self,
        controller: BaseAgentController,
        suite: TestSuite,
    ) -> List[TestResult]:
        """Run a test suite against a controller.

        Args:
            controller: The agent controller to test
            suite: The test suite to run

        Returns:
            List of test results
        """
        results = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running test suite: {suite.name}")
            print(f"{'='*60}\n")

        for test in suite.tests:
            result = self._run_single_test(controller, test)
            results.append(result)

            if self.verbose:
                print(result)

        # Print summary
        if self.verbose:
            self._print_summary(results)

        return results

    def _run_single_test(
        self,
        controller: BaseAgentController,
        test: Dict[str, Any],
    ) -> TestResult:
        """Run a single test case.

        Args:
            controller: The controller to test
            test: Test configuration

        Returns:
            TestResult
        """
        test_name = test["name"]
        input_data = test["input"]
        expected_output = test.get("expected_output")
        validator = test.get("validator")

        try:
            # Execute agent
            start_time = time.time()
            output = controller.invoke(input_data)
            latency_ms = (time.time() - start_time) * 1000

            # Validate output
            passed = True
            error = None

            if expected_output is not None:
                if output != expected_output:
                    passed = False
                    error = f"Expected {expected_output}, got {output}"

            if validator is not None:
                try:
                    if not validator(output):
                        passed = False
                        error = "Custom validator failed"
                except Exception as e:
                    passed = False
                    error = f"Validator error: {e}"

            return TestResult(
                test_name=test_name,
                passed=passed,
                latency_ms=latency_ms,
                error=error,
                output=output,
            )

        except Exception as e:
            # Test failed with exception
            return TestResult(
                test_name=test_name,
                passed=False,
                latency_ms=0.0,
                error=f"Exception: {e}",
                output=None,
            )

    def _print_summary(self, results: List[TestResult]):
        """Print test summary.

        Args:
            results: List of test results
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Test Summary:")
        print(f"  Total:   {total}")
        print(f"  Passed:  {passed}")
        print(f"  Failed:  {failed}")
        print(f"  Pass Rate: {passed/total*100:.1f}%" if total > 0 else "  Pass Rate: N/A")
        print(f"  Avg Latency: {avg_latency:.2f}ms")
        print(f"{'='*60}\n")
