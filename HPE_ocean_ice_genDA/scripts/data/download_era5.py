"""Submit regional daily ERA5 preprocessing from WeatherBench2."""

import sys

from dawgz import job, schedule
from functools import partial
from pathlib import Path

from appa2.config.hydra import compose
from appa2.data.download import download_daily_forcing


def main():
    config_path = Path(__file__).resolve().parent / "configs" / "download_era5.yaml"
    config = compose(config_path, overrides=sys.argv[1:])
    hardware = config.hardware
    process = partial(
        download_daily_forcing,
        output_dir=config.output_dir,
        start_date=config.start_date,
        end_date=config.end_date,
        south=config.south,
        north=config.north,
    )
    task = job(process, name="ERA5-daily", **hardware.job)
    options = {"export": "ALL"}
    if hardware.account is not None:
        options["account"] = hardware.account

    schedule(task, name="ERA5", backend=hardware.backend, **options)


if __name__ == "__main__":
    main()
