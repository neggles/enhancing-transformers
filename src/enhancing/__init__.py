try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from . import cli, dataloader, losses, modules, utils

__all__ = [
    "__version__",
    "version_tuple",
    "cli",
    "dataloader",
    "losses",
    "modules",
    "utils",
]
