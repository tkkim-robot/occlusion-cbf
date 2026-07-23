"""Compatibility entry point for the former ``crowd2`` scenario name.

The route-focused benchmark is now :mod:`examples.test_crowd`.
"""

try:
    from examples import test_crowd as _implementation
except ImportError:
    import test_crowd as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    _implementation.main()
