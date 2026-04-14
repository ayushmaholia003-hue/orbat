"""Run all tests for ORBAT classification system."""

import pytest
import sys

if __name__ == '__main__':
    # Run tests with verbose output
    exit_code = pytest.main([
        'tests/',
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    
    sys.exit(exit_code)
