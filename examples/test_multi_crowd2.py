"""Compatibility entry point for :mod:`examples.test_multi_crowd`."""

try:
    from examples import test_multi_crowd as _implementation
except ImportError:
    import test_multi_crowd as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    _implementation.main()
