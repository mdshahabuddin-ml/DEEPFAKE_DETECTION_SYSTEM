"""User Awareness Module - Educational content and best practices.

This module provides educational content about deepfakes, detector
limitations, and best practices for users.

Purpose:
    - Educate users about deepfake technology
    - Explain why detectors can make mistakes
    - Provide best practices for verification
    - Promote critical thinking about media authenticity
"""
__version__ = "0.1.0"

_LAZY_IMPORTS = {
    "EducationalContent": ("user_awareness.educational_content", "EducationalContent"),
    "LimitationsDocument": ("user_awareness.limitations_doc", "LimitationsDocument"),
    "BestPractices": ("user_awareness.best_practices", "BestPractices"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    """PEP 562 lazy attribute access."""
    import importlib

    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
