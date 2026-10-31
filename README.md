# Visible-evidence-guided virtual completion of Dunhuang murals

## Model weights

- **Author-released checkpoints (`pth.zip`):** [Baidu Netdisk](https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw) — access code: `mgi2`
- **Official pretrained starting weights:** [LaMa](https://github.com/advimman/lama) · [StrDiffusion](https://github.com/htyjers/StrDiffusion) · [PGRDiff](https://github.com/CZY-Code/PGRDiff)

## Repository layout and training configuration

The training entry point provided with this revision is `train_new.py`
(the locally supplied file `train_new (7).py` should be renamed to `train_new.py`).
It imports `options`, `models`, `data`, `utils` and `str_utils` from the source-package
root. The configuration available in the released package is:

`selected_package/selected_package/options/train/train_dunhuang_final.yml`

The entry point accepts `-opt` and `--launcher {none,pytorch}`. The paper reports
single-GPU, non-distributed training, so use `--launcher none`.

### Prepare the source package

```bash
git clone https://github.com/xinyu0833-bit/Visible-evidence-guided.git
cd Visible-evidence-guided

# Place the supplied training entry point in the package root alongside
# the data/, models/, options/ and str_utils/ modules, with the `utils` module
required by the source also available on Python's import path.
cp "/path/to/train_new.py" selected_package/selected_package/train_new.py

cd selected_package/selected_package
test -f train_new.py
test -f options/train/train_dunhuang_final.yml
python - <<'PY'
import importlib.util
missing = [m for m in ('options', 'models', 'data', 'utils', 'str_utils')
           if importlib.util.find_spec(m) is None]
print('Missing imported modules:', missing)
if missing:
    raise SystemExit('Install or supply the missing StrDiffusion modules first.')
PY
```

The uploaded materials do not contain all upstream training dependencies.
The original script imports `utils` directly, so a compatible `utils.py` or
`utils` package must be present before the training command can execute.

Use the Python/CUDA/PyTorch environment required by the released StrDiffusion
implementation. The following packages are also imported by the training and
evaluation code; this is a dependency starting point, not a pinned environment:

```bash
python -m pip install numpy pandas opencv-python scikit-image pillow pyyaml \
  torchvision tensorboard lpips
```

Install a compatible CUDA-enabled PyTorch build separately if not already present.

### Configure training data and starting weights

Edit `options/train/train_dunhuang_final.yml` to point to **existing local files**.
These are configuration keys, not additional command-line options:

| YAML key | Required value |
| --- | --- |
| `degradation.mask_root` | Directory of Dunhuang training masks |
| `datasets.train.dataroot_GT` | Dunhuang training reference-image directory |
| `datasets.train.prompt_path` | The prompt JSON matched to those training records |
| `path.pretrain_model_G` | Actual pretrained texture/model checkpoint |
| `path.pretrain_model_Gs` | Actual pretrained structure-network checkpoint |
| `path.resume_state` | `null` for a new run, or an existing training-state file to resume |
| `train.niter` | `20000` for the reported training budget |
| `datasets.train.batch_size` | `8` for the reported batch size |
| `name` | A distinct experiment identifier for each run |

The released root-level `DUNHUANG_train_prompts.json` is a published prompt file;
verify that its sample IDs correspond to the configured image records. Do not
assume that changing the JSON filename alone guarantees record alignment.

The released YAML currently contains original AutoDL absolute paths such as
`/root/autodl-tmp/...`; these must be changed for another machine. The training
script's own default `-opt` path also points to an older AutoDL installation,
so **always pass `-opt` explicitly**.

### Single-GPU training command (as implemented by the supplied script)

Run this **from `selected_package/selected_package`** after configuring the
dataset, prompts, checkpoints and experiment output paths:

```bash
python train_new.py \
  -opt options/train/train_dunhuang_final.yml \
  --launcher none
```

The script reads the training budget from `train.niter`, the data paths from
`datasets.train`/`degradation.mask_root`, the initial weights from `path`, and
the save interval from `logger.save_checkpoint_freq`. The released YAML sets
20,000 iterations, batch size 8, learning rate 1e-5 with reductions at
10,000/16,000 iterations, and a checkpoint interval of 2,000 iterations.
It also writes a final checkpoint via `model.save("latest")`.

**Important implementation caveat:** the supplied `train_new.py` builds an
independently shuffled `Datasetset_mask` / `DataLoader`, and advances that loader
separately from the image loader. It does **not**, as written, guarantee the
sample-ID pairing and synchronised image/mask transformations described in
the manuscript. The configuration also exposes a `manual_seed` field, but the
supplied script does not call the Python/NumPy/PyTorch RNG seeding functions.
The command above is the faithful invocation of the **supplied script**;
it should not be presented as a verified reproduction of every reported
training condition until the provenance and record pairing are reconciled.

For the paper's three reported runs (seeds `1234`, `2024`, `3407`), preserve
the actual run-specific configurations, logs and output checkpoints. Merely
changing the YAML seed value in this version of the script is not a
verified way to reproduce three distinct seeded experiments.

## Evaluation of saved predictions

The commands below evaluate **saved predictions**, not model training or
inference. Keep input image IDs, masks and predictions aligned for each set.

From the repository root:

```bash
cd /path/to/Visible-evidence-guided
python -m pip install numpy pandas opencv-python scikit-image torch lpips

EVAL_SCRIPT="$(find . -type f -name 'eval_paired_predictions_paper_documented.py' -print -quit)"
test -n "$EVAL_SCRIPT" || { echo "Evaluation script not found"; exit 1; }
```

### Dunhuang: 100 fixed reference-based test records

```bash
python "$EVAL_SCRIPT" \
  --pred-dir /path/to/dunhuang/predictions \
  --gt-dir /path/to/dunhuang/references \
  --mask-dir /path/to/dunhuang/masks \
  --outdir evaluation_results/dunhuang \
  --dataset-name Dunhuang --expected 100 \
  --mask-convention known_white --size 256 --device cuda \
  --canny-low 100 --canny-high 200 \
  --boundary-kernel 5 --edge-tolerance 1.0
```

The Dunhuang test set consists of 80 records in the `[0, 0.20)` missing-area
stratum and 20 in `[0.20, 0.40)`; none are excluded. Full-set metrics are the
arithmetic mean of the individual-image values, not a mean weighted by mask
area. After image-level rechecking, the reported Dunhuang G-PSNR values are
**27.7755 dB for PowerPaint** and **28.5253 dB for RAD**.

### MuralDH: 201 controlled reference-based records

```bash
python "$EVAL_SCRIPT" \
  --pred-dir /path/to/muraldh_controlled/predictions \
  --gt-dir /path/to/muraldh_controlled/references \
  --mask-dir /path/to/muraldh_controlled/masks \
  --outdir evaluation_results/muraldh_controlled \
  --dataset-name MuralDH-controlled --expected 201 \
  --mask-convention known_white --size 256 --device cuda \
  --canny-low 100 --canny-high 200 \
  --boundary-kernel 5 --edge-tolerance 1.0
```

The controlled MuralDH reference-image–mask records are fixed before inference.
These pairs are **not** physical before-and-after observations of a mural.

### No-reference image-quality evaluation (Table 6)

Table 6 uses **201 originally damaged images and 201 output images per method**
(`n=201` for each reported row). ARNIQA, BRISQUE and DBCNN are calculated
separately from the paired-reference evaluator. No executable command for that
three-metric no-reference pipeline is included here because the supplied files
do not identify a verified entry point and argument interface for it.

## Reproducibility records

For each experimental run, retain its configuration, output checkpoint
identifier, starting checkpoint, input image–mask record IDs, prompt JSON,
individual-image metric file and execution log. The supplied code and README
are useful reproduction materials; a complete third-party replay of the
reported experiments has not been independently demonstrated by these files.

All datasets and third-party checkpoints remain subject to their
respective release terms.
