#!/usr/bin/env python3
"""Wrapper script to patch beaker import issue and launch experiments."""

# Patch beaker.exceptions before any other imports
import beaker.exceptions
if not hasattr(beaker.exceptions, 'BeakerSecretNotFound'):
    beaker.exceptions.BeakerSecretNotFound = beaker.exceptions.SecretNotFound

# Now import and run the actual launch script
import sys
from pathlib import Path

# Add the parent directory to path so we can import the launch script
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main function
from launch_data_size_sweep import main

if __name__ == "__main__":
    main()

