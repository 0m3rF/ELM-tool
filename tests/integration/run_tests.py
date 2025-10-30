#!/usr/bin/env python
"""
Test runner script for ELM-tool integration tests.

This script provides a convenient way to run integration tests with various options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run ELM-tool integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py --all
  
  # Run only small dataset tests
  python run_tests.py --small
  
  # Run only large dataset tests
  python run_tests.py --large
  
  # Run specific database combination
  python run_tests.py --combination postgresql mysql
  
  # Run with coverage report
  python run_tests.py --all --coverage
  
  # Quick smoke test
  python run_tests.py --quick
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all integration tests'
    )
    
    parser.add_argument(
        '--small',
        action='store_true',
        help='Run small dataset tests only (5 rows)'
    )
    
    parser.add_argument(
        '--large',
        action='store_true',
        help='Run large dataset tests only (50,000 rows)'
    )
    
    parser.add_argument(
        '--integrity',
        action='store_true',
        help='Run data integrity tests only'
    )
    
    parser.add_argument(
        '--edge',
        action='store_true',
        help='Run edge case tests only'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a quick smoke test (single small dataset test)'
    )
    
    parser.add_argument(
        '--combination',
        nargs=2,
        metavar=('SOURCE', 'TARGET'),
        help='Run tests for specific database combination (e.g., postgresql mysql)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--show-output',
        '-s',
        action='store_true',
        help='Show print statements and progress'
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    base_cmd = ['pytest', 'tests/integration/test_db_copy_integration.py']
    
    # Add verbosity
    if args.verbose:
        base_cmd.append('-v')
    
    # Add show output
    if args.show_output:
        base_cmd.append('-s')
    
    # Add coverage
    if args.coverage:
        base_cmd.extend(['--cov=elm.core.copy', '--cov-report=html', '--cov-report=term'])
    
    # Determine which tests to run
    if args.quick:
        # Quick smoke test
        test_path = '::TestSmallDatasetCopy::test_copy_postgres_to_mysql_small'
        cmd = base_cmd + [test_path]
        return run_command(cmd, "Quick Smoke Test (PostgreSQL → MySQL, 5 rows)")
    
    elif args.combination:
        source, target = args.combination
        # Run both small and large tests for this combination
        small_test = f'::TestSmallDatasetCopy::test_copy_5_rows_between_databases[{source}-{target}]'
        large_test = f'::TestLargeDatasetCopy::test_copy_500k_rows_between_databases[{source}-{target}]'
        
        print(f"\nRunning tests for {source.upper()} → {target.upper()}")
        
        # Run small test
        cmd_small = base_cmd + [small_test]
        result_small = run_command(cmd_small, f"Small Dataset Test ({source} → {target})")
        
        # Run large test
        cmd_large = base_cmd + [large_test]
        result_large = run_command(cmd_large, f"Large Dataset Test ({source} → {target})")
        
        return max(result_small, result_large)
    
    elif args.small:
        test_path = '::TestSmallDatasetCopy'
        cmd = base_cmd + [test_path]
        return run_command(cmd, "Small Dataset Tests (5 rows)")
    
    elif args.large:
        test_path = '::TestLargeDatasetCopy'
        cmd = base_cmd + [test_path]
        return run_command(cmd, "Large Dataset Tests (50,000 rows)")
    
    elif args.integrity:
        test_path = '::TestDataIntegrity'
        cmd = base_cmd + [test_path]
        return run_command(cmd, "Data Integrity Tests")
    
    elif args.edge:
        test_path = '::TestEdgeCases'
        cmd = base_cmd + [test_path]
        return run_command(cmd, "Edge Case Tests")
    
    elif args.all:
        return run_command(base_cmd, "All Integration Tests")
    
    else:
        # No specific option selected, show help
        parser.print_help()
        print("\n" + "="*70)
        print("No test option selected. Please choose one of the options above.")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

