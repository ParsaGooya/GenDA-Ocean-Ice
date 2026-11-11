# GenDA: Pixi setup on HPE

Set up Pixi separately on each HPE machine: home and scratch are not shared
between machines. Project folders can live anywhere. On each machine, put the
package cache (shared across projects on that machine) and all installed
project environments under `/lus/scratch/PGooya/pixi/`.

1. **Check the new machine and its scratch mount.**

   ```bash
   hostname -s
   uname -m
   df -h /lus/scratch/PGooya
   test -w /lus/scratch/PGooya && echo "Scratch is writable"
   ```

   If the path is unavailable or unwritable, stop and confirm the assigned
   scratch location with HPE.

2. **Find or install Pixi.**

   ```bash
   command -v pixi
   ls -l "$HOME/.pixi/bin/pixi"
   ```

   If the home binary exists, try it before reinstalling. If neither check
   finds Pixi, use the [official installer](https://pixi.prefix.dev/latest/installation/):

   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

   Check `pixi --version` against the project's `requires-pixi` before
   installing dependencies.

3. **Create the scratch directories and set the cache.**

   ```bash
   mkdir -p /lus/scratch/PGooya/pixi/{cache,envs}
   export PATH="$HOME/.pixi/bin:$PATH"
   export PIXI_CACHE_DIR=/lus/scratch/PGooya/pixi/cache
   pixi --version
   ```

   Add the two `export` lines above to `~/.bashrc` (Bash) or `~/.zshrc`
   (Zsh) so they take effect each time you open a terminal. Skip any line
   that's already there. If Pixi is installed elsewhere, use its actual
   location in the `PATH` line. If a batch job doesn't read your shell
   settings, add the `PIXI_CACHE_DIR` line to the job script.

4. **Set detached environments and the default channel.**

   ```bash
   pixi config set --global detached-environments /lus/scratch/PGooya/pixi/envs
   pixi config set --global default-channels '["https://conda.anaconda.org/conda-forge"]'
   ```

   This applies to projects anywhere on the machine unless a project
   overrides the setting. Each project keeps its own installed environment
   in scratch and can reuse the machine's package cache.
   [Pixi configuration reference](https://pixi.prefix.dev/latest/reference/pixi_configuration/).

5. **Verify before installing.** From the project directory containing
   `pixi.toml`, run:

   ```bash
   printf '%s\n' "$PIXI_CACHE_DIR"
   pixi config list detached-environments
   pixi config list default-channels
   pixi workspace channel list --urls
   grep -E 'https?://' pixi.lock | sort -u
   ```

   Confirm the paths point to scratch and all Conda channels and locked Conda
   package URLs use `https://conda.anaconda.org/conda-forge/`. PyPI and official
   PyTorch URLs can also appear for Python dependencies. If you find
   `defaults`, `repo.anaconda.com`, or unexpected channels or mirrors, stop
   before installing. The default-channel setting is not a network block;
   machine-wide enforcement requires HPE network controls.

6. **Install a compatible environment and check its actual location.**
   For example, for Martin2025 on an ARM Linux host supported by its lockfile:

   ```bash
   cd /path/to/GenDA/Martin2025
   pixi install --locked
   pixi run --locked python -c 'import sys; print(sys.prefix)'
   pixi shell
   ```

   The Python path should start with `/lus/scratch/PGooya/pixi/envs/`.
   Replace `/path/to/GenDA` with your checkout location.
   For a named environment, add `-e ENV_NAME` to each Pixi command.
   if Pixi reports an old home-based environment, verify the scratch installation before cleaning up the old copy.

Keep `pixi.toml` and `pixi.lock` in Git. If scratch is cleared, recreate the
environment with `pixi install --locked`.
