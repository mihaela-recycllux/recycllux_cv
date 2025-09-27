#!/usr/bin/env python3
"""
Main entry point for the Plastic Detection System.
A modular satellite-based plastic detection workflow using Google Earth Engine.

This is now a lightweight entry point that delegates to the CLI module.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the CLI
from cli.main_cli import main

if __name__ == '__main__':
    sys.exit(main())