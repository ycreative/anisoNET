"""Core package for anisoNET.

The public manuscript name is ``anisoNET``. The Python package uses the
lowercase import name ``anisonet``.

Heavy scientific dependencies are imported from submodules rather than at
package import time. This keeps command-line workflows predictable across
Windows and Linux environments.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
