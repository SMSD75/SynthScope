"""``python -m agid`` -> same as ``python -m synthscope`` (deprecated alias)."""
from synthscope.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
