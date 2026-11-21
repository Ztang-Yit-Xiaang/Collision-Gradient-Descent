"""Run the local prototype without installing a package into the system Python."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from collision_optim.smoke import main

if __name__ == "__main__":
    raise SystemExit(main())

