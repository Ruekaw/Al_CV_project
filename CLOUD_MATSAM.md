# MatSAM Cloud Run Notes

## Local status

- Current machine: RTX 3050 Laptop, 4GB VRAM.
- Current Python env: Python 3.12, no `torch` installed.
- Local checkpoints exist:
  - `MatSAM/checkpoints/sam_vit_b_01ec64.pth`
  - `MatSAM/checkpoints/sam_vit_h_4b8939.pth`

## Recommended cloud GPU

- `vit_b`: start with 8GB VRAM, 12GB+ preferred.
- `vit_h`: use 24GB+ VRAM.

For this project, begin with `vit_b` and `matsam.points_per_batch: 64`.
If CUDA OOM appears, lower it to `32` or `16`.

## Cloud setup

If you upload the Git bundle, restore the project first:

```bash
git clone CVproject_cloud_<timestamp>.bundle CVproject
cd CVproject
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-cloud.txt
```

If checkpoints are not uploaded, download at least `vit_b`:

```bash
mkdir -p MatSAM/checkpoints
wget -O MatSAM/checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Probe command

```bash
python scripts/batch_run.py --names 1-1晶 --runs runs_matsam_cloud_probe
```

Expected signal: output line should show `method=matsam`.
If it shows `method=baseline`, read the preceding `[matsam] ...` message first.
