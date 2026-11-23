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
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path

from appa2.config import PATH_ERA5, PATH_STAT
from appa2.config.hydra import compose
from appa2.data.const import (
    CONTEXT_VARIABLES,
    STATE_VARIABLES,
    ATM_FORCING_VARIABLES,
    SURFACE_VARIABLES,
    OCEAN_VARIABLES,
)

from experiments.autoencoder.train import train
from appa2.data.data_config import get_dataset_specs
from appa2.data.dataloaders import get_dataloader
from appa2.data.datasets import ERA5Dataset
from appa2.data.transforms import StandardizeTransform
from appa2.date import split_interval
from appa2.save import load_auto_encoder


def dump_to_latent(config):
    num_chunks = config.num_chunks
    start_date = config.get("start_date")
    end_date = config.get("start_date")

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


    if (start_date is None or end_date is None):
        start_date, end_date = data_configs.time_splits[0]

    time_intervals = split_interval(num_chunks, start_date, end_date)

    @job(
        name="appa dump (chunk)",
        array=num_chunks,
        **hardware_cfg.latent_chunk,
    )
    @torch.no_grad()
    def dump_chunk(rank: int):
        dask.config.set(scheduler="synchronous")

        time_interval = time_intervals[rank]

        st_state = StandardizeTransform(
            data_configs.path_stat_state,
            state_variables=STATE_VARIABLES,
            context_variables=CONTEXT_VARIABLES if len(CONTEXT_VARIABLES)>0 else None,
            depths=data_configs.depth_levels,
            depth_dim_name=data_configs.depth_dim_name
        )

        # st_forcing = StandardizeTransform(
        #     data_configs.path_stat_forcing,
        #     state_variables=ATM_FORCING_VARIABLES,
        #     depths=data_configs.depth_levels,
            # depth_dim_name=data_configs.depth_dim_name
        # )

        dataset = data_configs.dataset(
                    state_variables_path=data_configs.state_variables_path,
                    start_date=time_interval[0],
                    end_date=time_interval[1],
                    transform=st_state,
                    surface_variables= SURFACE_VARIABLES,
                    depth_variables= OCEAN_VARIABLES,
                    forcing_variables= ATM_FORCING_VARIABLES,
                    context_variables= CONTEXT_VARIABLES,
                    forcing_path = data_configs.forcing_path,
                    # forcing_transform=st_forcing,
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

        len_dl = len(dataloader)

        print("Total dataset size:", len(dataset), flush=True)
        print("Local chunk size:", len_dl, flush=True)

        dump_tmp_file = h5py.File(output_path / f"tmp/{rank}.h5", "w")
        dump_tmp_file.create_dataset("latents", (len_dl, *latent_state_shape), "float32")
        dump_tmp_file.create_dataset("forcings", (len_dl, *latent_state_shape), "float32")
        dump_tmp_file.create_dataset("dates", (len_dl, 4), "int32")

        dumped_states = []
        dumped_forcings = []
        timestamps = []
        curr_idx = 0
        start_time = time.time()

        def save_progress():
            nonlocal curr_idx, dumped_states, dumped_forcings, timestamps, start_time

            dumped_states = np.concatenate(dumped_states, axis=0)
            dumped_forcings = np.concatenate(dumped_forcings, axis=0)
            timestamps = np.concatenate(timestamps, axis=0)
            dump_tmp_file["latents"][curr_idx : curr_idx + len(dumped_states)] = dumped_states
            dump_tmp_file["forcings"][curr_idx : curr_idx + len(dumped_forcings)] = dumped_forcings
            dump_tmp_file["dates"][curr_idx : curr_idx + len(dumped_states)] = timestamps
            curr_idx += len(dumped_states)

            dumped_states = []
            dumped_forcings = []
            timestamps = []
            gc.collect()

        for idx, batch in enumerate(dataloader):
            state, context, date, forcing, _, _ = train.prepare_batch(batch)

            z, f = autoencoder.encode(state, date, context, forcing)

            dumped_states.append(z.cpu().numpy())
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

    @after(dump_chunk)
    @job(
        name="appa dump (merge)",
        **hardware_cfg.aggregate,
    )
    def aggregate():

        dataset = data_configs.dataset(
                    state_variables_path=data_configs.state_variables_path,
                    start_date=start_date,
                    end_date=end_date,
                    surface_variables= SURFACE_VARIABLES,
                    depth_variables= OCEAN_VARIABLES,
                    forcing_variables= ATM_FORCING_VARIABLES,
                    context_variables= CONTEXT_VARIABLES,
                    forcing_path = data_configs.forcing_path,
                    depths=[],
                    fill_nans=False,
                )
        total_num_samples = len(dataset)

        # TODO: Function to compute shape from AE config alone.
        autoencoder = load_auto_encoder(output_path / "ae", model_name="model")
        latent_state_shape = autoencoder.latent_shape
        if len(latent_state_shape) == 3:
            latent_state_shape = (
                latent_state_shape[0] * latent_state_shape[1],
                latent_state_shape[2],
            )
        del autoencoder

        dump_tmp_file = h5py.File(output_path / config.dump_name, "w")
        dump_tmp_file.create_dataset(
            "latents", (total_num_samples, *latent_state_shape), "float32"
        )
        dump_tmp_file.create_dataset(
            "forcings", (total_num_samples, *latent_state_shape), "float32"
        )
        dump_tmp_file.create_dataset("dates", (total_num_samples, 4), "int32")

        curr_idx = 0

        for rank in range(num_chunks):
            dump_tmp_file_chunk = h5py.File(output_path / f"tmp/{rank}.h5", "r")

            latents = dump_tmp_file_chunk["latents"][:]
            forcings = dump_tmp_file_chunk["forcings"][:]
            dates = dump_tmp_file_chunk["dates"][:]

            print(f"Chunk {rank} shape: {latents.shape}, {forcings.shape}, {dates.shape}", flush=True)

            num_samples_chunk = latents.shape[0]

            # Truncate last chunk, if necessary.
            num_samples_chunk = min(num_samples_chunk, total_num_samples - curr_idx)
            latents = latents[:num_samples_chunk]
            forcings = forcings[:num_samples_chunk]
            dates = dates[:num_samples_chunk]

            dump_tmp_file["latents"][curr_idx : curr_idx + num_samples_chunk] = latents
            dump_tmp_file["forcings"][curr_idx : curr_idx + num_samples_chunk] = forcings
            dump_tmp_file["dates"][curr_idx : curr_idx + num_samples_chunk] = dates
            curr_idx += num_samples_chunk

        print(dump_tmp_file["latents"].shape, dump_tmp_file["forcings"].shape, dump_tmp_file["dates"].shape, flush=True)

        dump_tmp_file.close()

        shutil.rmtree(output_path / "tmp")

    schedule(
        aggregate,
        name="appa dump",
        export="ALL",
        account=config.hardware.account,
        backend=config.hardware.backend,
    )


if __name__ == "__main__":
    config = compose("configs/dump.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" not in config:
        config["id"] = wandb.util.generate_id()

    dump_to_latent(config)
