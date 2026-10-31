# Visible-evidence-guided virtual completion of Dunhuang murals

## Model weights

- **Author-released checkpoints (`pth.zip`):** [Baidu Netdisk](https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw) — access code: `mgi2`
- **Official pretrained starting weights:** [LaMa](https://github.com/advimman/lama) · [StrDiffusion](https://github.com/htyjers/StrDiffusion) · [PGRDiff](https://github.com/CZY-Code/PGRDiff)

## Reproduction commands

```bash
git clone https://github.com/xinyu0833-bit/Visible-evidence-guided.git
cd Visible-evidence-guided
python -m pip install numpy pandas opencv-python scikit-image torch lpips

# Locate the evaluation script in the released source package.
EVAL_SCRIPT="$(find . -type f -name 'eval_paired_predictions_paper_documented.py' -print -quit)"
echo "$EVAL_SCRIPT"
```

### Dunhuang (100 reference-based test records)

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

### MuralDH (201 controlled reference-based records)

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

The commands above evaluate **saved model predictions**; replace the input paths with the matching released records and outputs. Training and inference require their corresponding scripts and configurations in the released code package.
