#!/usr/bin/env python3
"""
Simple test runner that doesn't require pytest
Runs all tests and reports results
"""
import sys
import importlib.util
from pathlib import Path


def run_test_file(filepath):
    """Run tests from a single file"""
    print(f"\n{'='*70}")
    print(f"Running {filepath.name}")
    print('='*70)

    # Load the module
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find all test classes
    test_classes = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith('Test') and isinstance(getattr(module, name), type)
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print('-'*70)

        # Find all test methods
        test_methods = [
            name for name in dir(test_class)
            if name.startswith('test_') and callable(getattr(test_class, name))
        ]

        for method_name in test_methods:
            total_tests += 1
            test_instance = test_class()

            # Run setup if exists
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()

            try:
                # Run the test
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                failed_tests += 1

            # Run teardown if exists
            if hasattr(test_instance, 'teardown_method'):
                try:
                    test_instance.teardown_method()
                except:
                    pass

    return total_tests, passed_tests, failed_tests


def main():
    """Run all tests"""
    print("="*70)
    print("LLM Abuse Patterns - Test Suite")
    print("="*70)

    tests_dir = Path(__file__).parent / 'tests'
    test_files = sorted(tests_dir.glob('test_*.py'))

    if not test_files:
        print("No test files found!")
        return 1

    total = 0
    passed = 0
    failed = 0

    for test_file in test_files:
        t, p, f = run_test_file(test_file)
        total += t
        passed += p
        failed += f

    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print('='*70)
    print(f"Total tests:  {total}")
    print(f"Passed:       {passed} ✓")
    print(f"Failed:       {failed} ✗")
    print(f"Success rate: {(passed/total*100):.1f}%")
    print('='*70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
