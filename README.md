# Visible-evidence-guided virtual completion of Dunhuang murals

Reproducibility resources for the manuscript submitted to *npj Heritage Science*. The code and manifests document the experimental workflow; evaluating saved predictions and independently reproducing the full training run are distinct operations.

## Model weights

### Project-released checkpoints

- Released checkpoints (`pth.zip`): https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw (access code: `mgi2`)

### Official starting weights for matched-training baselines

LaMa, StrDiffusion and PGRDiff were **fine-tuned from officially released pretrained weights, not trained from scratch**, on the matched Dunhuang training records. The starting-weight variants actually used were:

| Baseline | Official starting-weight variant | Official project |
| --- | --- | --- |
| LaMa | `big-lama` | [LaMa](https://github.com/advimman/lama) |
| StrDiffusion | `PSV` pretrained weights | [StrDiffusion](https://github.com/htyjers/StrDiffusion) |
| PGRDiff | `DUNHUANG` pretrained checkpoint | [PGRDiff](https://github.com/CZY-Code/PGRDiff) |

These are the **initialization variants for the three fine-tuned baselines**; they should not be confused with the resulting fine-tuned checkpoints or with the frozen-backbone setup of the proposed method described in the manuscript.

## Code and experiment-file inventory

All paths below are relative to the repository root, unless noted otherwise.

| Resource | File or location | Purpose |
| --- | --- | --- |
| Dunhuang training prompts | [`DUNHUANG_train_prompts.json`](DUNHUANG_train_prompts.json) | Fixed training prompts for the reported matched conditions. |
| Dunhuang test prompts | [`DUNHUANG_test_prompts.json`](DUNHUANG_test_prompts.json) | Prompts for the independent 100-image Dunhuang test set. |
| Controlled MuralDH test prompts | [`MuralDH_test_prompts.json`](MuralDH_test_prompts.json) | Prompts for the controlled 201-record MuralDH reference-based evaluation. |
| Controlled MuralDH pairing manifest | [`selected_package/selected_package/dataset/pairs_manifest.csv`](selected_package/selected_package/dataset/pairs_manifest.csv) | Deterministic image-mask mapping, original IDs, hashes and missing-area fractions. |
| MuralDH pair preparation | [`selected_package/selected_package/dataset/muraldh_prepare_201.py`](selected_package/selected_package/dataset/muraldh_prepare_201.py) | Documents selection and preparation of the controlled pairs. |
| Prompt-generation procedure | [`selected_package/selected_package/dataset/generate_prompts_qwen.py`](selected_package/selected_package/dataset/generate_prompts_qwen.py) | Qwen2-VL prompt-generation workflow. |
| Training entry point | `selected_package/selected_package/dataset/train_new.py` | Training script; see the command and configuration below. |
| Final training configuration | [`selected_package/selected_package/options/train/train_dunhuang_final.yml`](selected_package/selected_package/options/train/train_dunhuang_final.yml) | Final-model training configuration; update machine-specific paths before use. |
| Paired-prediction evaluation | `eval_paired_predictions_paper_documented.py` (locate with the command below) | Recalculate reference-based metrics from aligned saved prediction, reference and mask files. |

## Training

The project training entry point is documented as `dataset/train_new.py` under `selected_package/selected_package/`. Run the following command from that package directory, after setting local dataset and checkpoint paths in the YAML configuration. This command describes the training workflow; the full experiment also requires the stated datasets and checkpoints.

```bash
git clone https://github.com/xinyu0833-bit/Visible-evidence-guided.git
cd Visible-evidence-guided/selected_package/selected_package

python dataset/train_new.py -opt options/train/train_dunhuang_final.yml --launcher none
```

## Testing: recalculating metrics from saved predictions

The commands below evaluate existing prediction images; they do not launch model inference or retraining. Use matching prediction, reference and binary-mask sample IDs. Run from the repository root:

```bash
cd /path/to/Visible-evidence-guided
EVAL_SCRIPT="$(find . -type f -name 'eval_paired_predictions_paper_documented.py' -print -quit)"
test -n "$EVAL_SCRIPT" || { echo "Evaluation script not found; check the extracted source package." >&2; exit 1; }

# Dunhuang: 100 test cases
python "$EVAL_SCRIPT" \
  --pred-dir /path/to/dunhuang/predictions \
  --gt-dir /path/to/dunhuang/references \
  --mask-dir /path/to/dunhuang/masks \
  --outdir evaluation_results/dunhuang \
  --dataset-name Dunhuang --expected 100 \
  --mask-convention known_white --size 256 --device cuda \
  --canny-low 100 --canny-high 200 \
  --boundary-kernel 5 --edge-tolerance 1.0

# MuralDH: 201 controlled evaluation cases
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


The controlled MuralDH experiment uses independently paired intact reference images and real-deterioration masks; it is **not** an evaluation against the historical pre-damage appearance of the same murals. Other no-reference scores on originally damaged images are reported separately in the manuscript.
