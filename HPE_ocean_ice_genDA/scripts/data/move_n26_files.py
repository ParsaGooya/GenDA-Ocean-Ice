"""Move NetCDF files with exact latitude bounds [26, 90] into N26."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/lus/scratch/PGooya/data/glorys_daily")
    )
    parser.add_argument("--dry-run", action="store_true", help="List moves without moving files.")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    destination = root / "N26"
    moves = []
    errors = []

    # Prune N26 before recursion, including on subsequent runs.
    import os

    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        subdirectories[:] = sorted(
            name for name in subdirectories
            if Path(directory) / name != destination
        )
        for filename in sorted(filenames):
            source = Path(directory) / filename
            if source.suffix.lower() not in {".nc", ".nc4", ".netcdf"} or source.is_symlink():
                continue
            try:
                with xr.open_dataset(source, chunks=None, cache=False) as ds:
                    coordinate = next((name for name in ("latitude", "lat") if name in ds), None)
                    if coordinate is None:
                        print(f"SKIP (no latitude coordinate): {source}")
                        continue
                    latitude = np.asarray(ds[coordinate].values)
                    matches = (
                        latitude.size > 0
                        and np.isfinite(latitude).all()
                        and latitude.min() == 26.0
                        and latitude.max() == 90.0
                    )
                if matches:
                    moves.append((source, destination / source.relative_to(root)))
            except Exception as error:
                errors.append(f"{source}: {error}")

    # Validate the complete plan before moving any files.
    for source, target in moves:
        if target.exists() or target.is_symlink():
            errors.append(f"Destination already exists: {target}")
    if errors:
        raise SystemExit("No files moved. Resolve these errors:\n" + "\n".join(errors))

    for source, target in moves:
        print(f"{'WOULD MOVE' if args.dry_run else 'MOVE'}: {source} -> {target}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    print(f"{'Matched' if args.dry_run else 'Moved'} {len(moves)} file(s).")


if __name__ == "__main__":
    main()
