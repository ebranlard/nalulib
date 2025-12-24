import os
import sys
import types
import matplotlib

# Determine whether we have a suitable display
#print('>>> DISPLAY:',  os.environ.get('DISPLAY'))
#print('>>> backend:', matplotlib.get_backend().lower())

is_headless = matplotlib.get_backend().lower() == 'agg'

if is_headless:
    import nalulib.pyplot_term as _plt

else:
    import matplotlib.pyplot as _plt


#sys.modules[__name__] = _plt

# Create a proxy module to add "is_headless"
_proxy = types.ModuleType(__name__)
_proxy.__dict__.update(_plt.__dict__)  # Populate with all attributes of _plt
_proxy.is_headless = is_headless      # Add your custom flag

# Replace this module with the proxy
sys.modules[__name__] = _proxy


