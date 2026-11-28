r"""Encode ERA5 into latent space with an AE."""

import dask
import gc
import h5py
import numpy as np
import shutil
import sys
import time
import torch
import wandb

from dawgz import after, job, schedule
from datetime import date, timedelta
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path

from appa2.config.hydra import compose
from appa2.data.const import (
    CONTEXT_VARIABLES,
    STATE_VARIABLES,
    ATM_FORCING_VARIABLES,
    SURFACE_VARIABLES,
    OCEAN_VARIABLES,
)


from appa2.data.data_config import get_dataset_specs
from appa2.data.dataloaders import get_dataloader
from appa2.data.transforms import StandardizeTransform
from appa2.date import split_daily_interval
from appa2.save import load_auto_encoder


def dump_to_latent(config):
    start_date = config.get("start_date")
    end_date = config.get("end_date")

    model_path = Path(config.model_path)
    use_best = config.checkpoint == "best"

    save_every = config.save_every
    log_every = config.log_every

    output_path = model_path / "latents" / config.id
    (output_path / "tmp").mkdir(parents=True, exist_ok=True)

    # Copy AE
    (output_path / "ae").mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path / "config.yaml", output_path / "ae" / "config.yaml")
    shutil.copy2(
        model_path / ("model_best.pth" if use_best else "model_last.pth"),
        output_path / "ae" / "model.pth",
    )
    train_config =  compose(model_path / "config.yaml")
    data_configs = get_dataset_specs(train_config.grids.main) 
    hardware_cfg = config.hardware

    has_forcing = train_config.grids.get("forcing", None) is not None
    FORCING_VARIABLES = ATM_FORCING_VARIABLES if has_forcing else []

    train_start, train_end = data_configs.time_splits[0]
    if start_date is None:
        start_date = train_start
    if end_date is None:
        end_date = train_end

    time_intervals = split_daily_interval(config.num_chunks, start_date, end_date)
    num_chunks = len(time_intervals)

    @job(
        name="appa dump (chunk)",
        array=num_chunks,
        **hardware_cfg.latent_chunk,
    )
    @torch.no_grad()
    def dump_chunk(rank: int):
        dask.config.set(scheduler="synchronous")

        chunk_start, chunk_end = time_intervals[rank]

        st_state = StandardizeTransform(
            data_configs.path_stat_state,
            state_variables=STATE_VARIABLES,
            forcing_variables=FORCING_VARIABLES if len(FORCING_VARIABLES)>0 else None,
            depths=data_configs.depth_levels,
            depth_dim_name=data_configs.depth_dim_name
        )


        dataset = data_configs.dataset(
                    state_variables_path=data_configs.state_variables_path,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    trajectory_size=1,
                    transform=st_state,
                    surface_variables= SURFACE_VARIABLES,
                    depth_variables= OCEAN_VARIABLES,
                    forcing_variables= FORCING_VARIABLES,
                    context_variables= CONTEXT_VARIABLES,
                    forcing_path = data_configs.forcing_path,
                    context_path=data_configs.context_path,
                    depths=data_configs.depth_levels,
                    fill_nans=False,
                )

        dataloader = get_dataloader(
            dataset,
            batch_size=config.batch_size,
            num_workers=8,
            prefetch_factor=1,
            shuffle=False,
        )

        autoencoder = load_auto_encoder(output_path / "ae", model_name="model")
        autoencoder.cuda()
        autoencoder.eval()

        latent_state_shape = autoencoder.latent_shape
        if len(latent_state_shape) == 3:
            latent_state_shape = (
                latent_state_shape[0] * latent_state_shape[1],
                latent_state_shape[2],
            )

        num_rows = len(dataset)
        len_dl = len(dataloader)

        print("Total dataset size:", len(dataset), flush=True)
        print("Local chunk size:", len_dl, flush=True)

        dump_tmp_file = h5py.File(output_path / f"tmp/{rank}.h5", "w")
        dump_tmp_file.create_dataset("latents", (num_rows, *latent_state_shape), "float32")
        dump_tmp_file.create_dataset("dates", (num_rows, 4), "int32")

        dumped_states = []
        dumped_forcings = []
        timestamps = []
        curr_idx = 0
        start_time = time.time()

        def save_progress():
            nonlocal curr_idx, dumped_states, dumped_forcings, timestamps, start_time

            if not dumped_states:
                return

            dumped_states = np.concatenate(dumped_states, axis=0)
            if has_forcing:
                dumped_forcings = np.concatenate(dumped_forcings, axis=0)
            timestamps = np.concatenate(timestamps, axis=0)
            dump_tmp_file["latents"][curr_idx : curr_idx + len(dumped_states)] = dumped_states
            if has_forcing:
                dump_tmp_file["forcings"][curr_idx : curr_idx + len(dumped_forcings)] = dumped_forcings
            dump_tmp_file["dates"][curr_idx : curr_idx + len(dumped_states)] = timestamps
            curr_idx += len(dumped_states)

            dumped_states = []
            dumped_forcings = []
            timestamps = []
            gc.collect()


        def prepare_batch(batch: tuple[torch.Tensor, ...]):
            if len(batch) == 3:
                state, context, timestamp = batch
                forcing = None
            elif len(batch) == 4:
                state, context, timestamp, forcing = batch
            else:
                raise ValueError(f"Expected 3 or 4 batch tensors, got {len(batch)}")

            protocol = data_configs.rearange_protocol
            target = rearrange(state.to("cuda", non_blocking=True), protocol)
            context = rearrange(context.to("cuda", non_blocking=True), protocol)
            timestamp = rearrange(
                timestamp.to("cuda", non_blocking=True), "B T D -> (B T) D"
            )

            valid = torch.isfinite(target)
            state = torch.where(valid, target, 0.0)
            context = torch.where(torch.isfinite(context), context, 0.0)

            if forcing is not None:
                forcing = rearrange(forcing.to("cuda", non_blocking=True), protocol)
                forcing = torch.where(torch.isfinite(forcing), forcing, 0.0)

            return state, context, timestamp, forcing

        for idx, batch in enumerate(dataloader):
            state, context, date, forcing = prepare_batch(batch)

            z, ff = autoencoder.encode(state, 
                                      date, 
                                      context, 
                                      forcing, 
                                      return_forcing_features=True)

            dumped_states.append(z.cpu().numpy())
            if has_forcing:
                # The final forcing features are on the latent mesh.
                f = ff[-1]
                if idx == 0:
                    dump_tmp_file.create_dataset("forcings", (num_rows, *f.shape[1:]), "float32")
                dumped_forcings.append(f.cpu().numpy())
            timestamps.append(date.cpu().numpy())

            if (idx + 1) % log_every == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"{idx + 1}/{len_dl} - Avg. time per sample: {(elapsed_time / log_every):.2f}s",
                    flush=True,
                )
                start_time = time.time()

            if len(dumped_states) >= save_every:
                save_progress()

        save_progress()
        dump_tmp_file.close()
        if curr_idx != num_rows:
            raise ValueError(f"Shard {rank} expected {num_rows} daily rows but wrote {curr_idx}.")

    @after(dump_chunk)
    @job(
        name="appa dump (merge)",
        **hardware_cfg.aggregate,
    )
    def aggregate():
        fields = {"latents": "float32", "dates": "int32"}
        if has_forcing:
            fields["forcings"] = "float32"

        shard_rows = []
        sample_shapes = {}
        for rank, (chunk_start, chunk_end) in enumerate(time_intervals):
            first_day = date.fromisoformat(chunk_start)
            last_day = date.fromisoformat(chunk_end)
            expected_rows = (last_day - first_day).days + 1
            expected_dates = []
            for offset in range(expected_rows):
                day = first_day + timedelta(days=offset)
                expected_dates.append([day.year, day.month, day.day, 0])

            with h5py.File(output_path / f"tmp/{rank}.h5", "r") as shard:
                for key in fields:
                    field = shard[key]
                    if field.ndim == 0 or field.shape[0] != expected_rows:
                        raise ValueError(
                            f"Shard {rank} expected {expected_rows} rows for {key}, "
                            f"but found shape {field.shape}."
                        )
                    shape = field.shape[1:]
                    expected_shape = sample_shapes.setdefault(key, shape)
                    if shape != expected_shape:
                        raise ValueError(
                            f"Shard {rank} expected {key} row shape {expected_shape}, "
                            f"but found {shape}."
                        )

                if not np.array_equal(shard["dates"][:], expected_dates):
                    raise ValueError(
                        f"Shard {rank} dates do not match {chunk_start} through {chunk_end}."
                    )
                shard_rows.append(shard["latents"].shape[0])

        final_path = output_path / config.dump_name
        partial_path = final_path.with_name(final_path.name + ".partial")
        total_rows = sum(shard_rows)
        with h5py.File(partial_path, "w") as merged:
            for key, dtype in fields.items():
                merged.create_dataset(key, (total_rows, *sample_shapes[key]), dtype=dtype)

            offset = 0
            for rank, rows in enumerate(shard_rows):
                with h5py.File(output_path / f"tmp/{rank}.h5", "r") as shard:
                    for start in range(0, rows, config.batch_size):
                        stop = min(start + config.batch_size, rows)
                        for key in fields:
                            merged[key][offset + start : offset + stop] = shard[key][start:stop]
                offset += rows

        partial_path.replace(final_path)
        shutil.rmtree(output_path / "tmp")

    options = {"export": "ALL"}
    if config.hardware.get("account") is not None:
        options["account"] = config.hardware["account"]

    schedule(
        aggregate,
        name="appa dump",
        backend=config.hardware.backend,
        **options
    )


if __name__ == "__main__":
    config = compose("configs/dump.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" not in config:
        config["id"] = wandb.util.generate_id()

    dump_to_latent(config)
