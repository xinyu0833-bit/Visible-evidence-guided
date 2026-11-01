# Visible-evidence-guided virtual completion of Dunhuang murals

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

## Training

The training script is located at `dataset/train_new.py` under `selected_package/selected_package/`. Run the following command from that package directory; update local dataset and checkpoint paths in the YAML configuration before execution.

```bash
git clone https://github.com/xinyu0833-bit/Visible-evidence-guided.git
cd Visible-evidence-guided/selected_package/selected_package

python dataset/train_new.py -opt options/train/train_dunhuang_final.yml --launcher none
```

## Testing (evaluation of saved predictions)

Run from the repository root:

```bash
cd /path/to/Visible-evidence-guided
EVAL_SCRIPT="$(find . -type f -name 'eval_paired_predictions_paper_documented.py' -print -quit)"

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
