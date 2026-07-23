"""Compatibility entry point for the former ``crowd1`` scenario name.

The legacy small workspace is now :mod:`examples.test_crowd_narrow`.
"""

try:
    from examples import test_crowd_narrow as _implementation
except ImportError:
    import test_crowd_narrow as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    _implementation.main()
