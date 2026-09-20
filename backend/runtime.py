"""Runtime capability flags.

Central place for optional-dependency detection so the app can boot in a
degraded "lite" mode on lightweight/serverless platforms where the heavy
RAG-Anything stack (MinerU, torch) is not installed.
"""

try:
    import raganything  # noqa: F401

    RAGANYTHING_AVAILABLE = True
except ImportError:
    RAGANYTHING_AVAILABLE = False
