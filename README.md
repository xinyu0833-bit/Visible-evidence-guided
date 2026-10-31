# Visible-evidence-guided virtual completion of Dunhuang murals

## Model weights

- Released checkpoints (`pth.zip`): https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw (access code: `mgi2`)

  

## Training

```bash
git clone https://github.com/xinyu0833-bit/Visible-evidence-guided.git
cd Visible-evidence-guided/selected_package/selected_package

python train_new.py -opt options/train/train_dunhuang_final.yml --launcher none
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
