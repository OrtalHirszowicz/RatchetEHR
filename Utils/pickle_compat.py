"""
Pickle compatibility utilities for loading sparse tensors across PyTorch versions.

Usage in Jupyter notebook or script:
    from Utils.pickle_compat import enable_compatibility
    enable_compatibility()
    
    # Now you can load old pickle files
    with open('path/to/file', 'rb') as f:
        data = pickle.load(f)
"""

import torch
import torch._utils


def enable_compatibility():
    """
    Enable compatibility mode for loading pickle files with sparse tensors
    created with different PyTorch versions.
    
    Call this once at the beginning of your script or notebook.
    """
    # Store original if it exists
    if hasattr(torch._utils, '_rebuild_sparse_tensor'):
        original_fn = torch._utils._rebuild_sparse_tensor
        
        def patched_rebuild_sparse_tensor(*args, **kwargs):
            # Handle the case where args has wrong count
            try:
                return original_fn(*args, **kwargs)
            except (ValueError, TypeError) as e:
                if "too many values to unpack" in str(e):
                    # Old format: (layout, data) where data is (indices, values, size, ...)
                    if len(args) == 2:
                        layout, data = args
                        # Extract indices, values, size from data
                        if isinstance(data, (tuple, list)) and len(data) >= 3:
                            indices = data[0]
                            values = data[1]
                            size = data[2]
                            # Build sparse tensor manually
                            return torch.sparse_coo_tensor(indices, values, size).coalesce()
                    # New format with more args
                    elif len(args) >= 4:
                        # Could be: layout, indices, values, size
                        try:
                            indices = args[1]
                            values = args[2]
                            size = args[3]
                            return torch.sparse_coo_tensor(indices, values, size).coalesce()
                        except:
                            pass
                raise e
        
        torch._utils._rebuild_sparse_tensor = patched_rebuild_sparse_tensor
        print("✓ PyTorch pickle compatibility mode enabled")
    else:
        print("⚠ Warning: _rebuild_sparse_tensor not found in torch._utils")


# Auto-enable on import if desired
# enable_compatibility()

