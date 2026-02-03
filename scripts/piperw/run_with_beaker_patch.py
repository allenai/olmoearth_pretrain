#!/usr/bin/env python3
"""Wrapper to patch beaker imports for compatibility with olmo_core."""

import sys

# Import and patch beaker BEFORE any other imports that might use it
import beaker
import beaker.exceptions

# Patch beaker.exceptions - create aliases for Beaker-prefixed exception names
_exception_aliases = {
    'BeakerSecretNotFound': 'SecretNotFound',
    'BeakerDatasetConflict': 'DatasetConflict',
    'BeakerDatasetNotFound': 'DatasetNotFound',
    'BeakerImageNotFound': 'ImageNotFound',
}

for alias_name, real_name in _exception_aliases.items():
    if not hasattr(beaker.exceptions, alias_name) and hasattr(beaker.exceptions, real_name):
        setattr(beaker.exceptions, alias_name, getattr(beaker.exceptions, real_name))

# Patch beaker main module - create aliases for Beaker-prefixed classes
# Map new names to old Beaker-prefixed names that olmo_core expects
_beaker_aliases = {
    'BeakerDataset': 'Dataset',
    'BeakerExperiment': 'Experiment', 
    'BeakerExperimentSpec': 'ExperimentSpec',
    'BeakerJob': 'Job',
    'BeakerJobPriority': 'Priority',  # Priority is the actual name in newer beaker
    'BeakerRetrySpec': 'RetrySpec',
    'BeakerTaskResources': 'TaskResources',
    'BeakerTaskSpec': 'TaskSpec',
    'BeakerDatasetConflict': 'DatasetConflict',
    'BeakerDatasetNotFound': 'DatasetNotFound',
}

for alias_name, real_name in _beaker_aliases.items():
    if not hasattr(beaker, alias_name) and hasattr(beaker, real_name):
        setattr(beaker, alias_name, getattr(beaker, real_name))

# Update sys.modules to ensure the patched version is used
sys.modules['beaker'] = beaker
sys.modules['beaker.exceptions'] = beaker.exceptions

# Patch olmo_core's get_beaker_username to handle missing user_name attribute
def patch_olmo_core_username():
    """Patch olmo_core to handle beaker API changes for getting username."""
    try:
        import olmo_core.internal.common as olmo_common
        original_get_beaker_username = olmo_common.get_beaker_username
        
        def patched_get_beaker_username():
            beaker_client = olmo_common.get_beaker_client()
            if beaker_client is None:
                return None
            # Try the old API first
            if hasattr(beaker_client, 'user_name'):
                return beaker_client.user_name
            # Try getting from account API
            try:
                # Try to get current user from account
                account_info = beaker_client.account.get(beaker_client.config.default_org)
                if hasattr(account_info, 'name'):
                    return account_info.name
                if hasattr(account_info, 'username'):
                    return account_info.username
            except Exception:
                pass
            # Fallback: try to extract from workspace or config
            if hasattr(beaker_client, 'config') and hasattr(beaker_client.config, 'default_workspace'):
                workspace = beaker_client.config.default_workspace
                if workspace and '/' in workspace:
                    # Extract username from workspace like "ai2/username-workspace" or "username/workspace"
                    parts = workspace.split('/')
                    if len(parts) >= 2:
                        org_or_user = parts[1].split('-')[0]  # Get first part before dash
                        return org_or_user
            return None
        
        olmo_common.get_beaker_username = patched_get_beaker_username
    except Exception:
        pass  # If patching fails, continue anyway

# Apply the patch
patch_olmo_core_username()

# Now import and run the target script
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    # The script path is the first argument
    script_path = Path(sys.argv[1]).resolve()
    # Change to the script's directory so relative imports work
    script_dir = script_path.parent
    os.chdir(script_dir)
    # Update sys.path to include the script's directory
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    # Remove the wrapper script and script path from argv, keep the rest
    sys.argv = [str(script_path.name)] + sys.argv[2:]
    # Execute the script
    with open(script_path) as f:
        code = compile(f.read(), str(script_path), 'exec')
        exec(code, {'__name__': '__main__', '__file__': str(script_path)})

