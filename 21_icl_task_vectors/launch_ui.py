#!/usr/bin/env python3
"""
Launcher script for ICL Task Vectors UI
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the UI
from ui.phase1_main import main

if __name__ == "__main__":
    main()
