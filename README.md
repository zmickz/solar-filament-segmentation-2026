# Solar filament segmentation — native-resolution U-Net ensemble with a metric-tuned instance rule

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(GONG H-alpha frames, MAGFiLO annotations).

| | |
|---|---|
| Out-of-fold Panoptic Quality (707 training frames) | **0.424** (95% bootstrap interval 0.415–0.433) |
| Public leaderboard | **0.37** |
| Model | 5 × U-Net, ResNet-34 encoder, 24 M parameters each, 49 MB per weight file |
| Training | 2 h 22 min on 2 × Tesla T4 (all five folds, including predictions) |
| Inference | about 2.6 s per frame on one T4 (five models, four flips each) |

## How it works

1. **Pre-processing.** Each 2048 × 2048 frame is standardised with the mean and standard deviation of the
   solar disk. No resizing.
2. **Network.** Five U-Nets (one per date-grouped fold) trained on 640-pixel crops. The target of a pixel is
   the *fraction of annotators* that marked it; the loss is binary cross-entropy + Dice.
3. **Inference.** Full frame in one pass, four flips, mean of the five models.
4. **Instances.** 8-connected regions of `p > 0.25` (disjoint by construction). A region is kept when its area is
   at least 320 px and its mean probability at least 0.85. These numbers were chosen by exhaustive search on
   out-of-fold maps, scored with the competition's pooled Panoptic Quality.

The technical report is in [`report/`](report/).

## Reproduce the submission (no training)

```bash
pip install -r requirements.txt
python scripts/predict.py --images <MAGFiLO>/test/test_images --out submission.csv
```

or open [`demo.ipynb`](demo.ipynb), which runs the same pipeline step by step, compares the result with the
file we submitted (`artifacts/submission_kaggle.csv`) and shows a few segmentations.
The weights are in `weights/` and are loaded by both. A GPU is recommended (CPU: about one minute per frame;
use `--limit 3` or `SFS_LIMIT=3` for a quick look).

## Reproduce the validation numbers

```bash
python scripts/evaluate_oof.py --annotations <MAGFiLO>/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
```

scores `artifacts/oof_maps.zip` (every training frame predicted by the fold model that never saw it) and
prints the pooled PQ, its bootstrap interval, per-fold values, IoU/Dice distribution and fragmentation counts.
Runs in about a minute on CPU.

## Retrain

`training/train_kaggle.py` is the exact script that produced the weights. It is a single file meant for a Kaggle
notebook with the competition data attached and two T4 GPUs: it builds a cache, rehearses every stage on a few
frames, trains the five folds (40 minutes each), predicts, tunes the rule and writes the submission.
`artifacts/training_log_kaggle.log` is the log of that run and `artifacts/folds.json` the split.

## Repository layout

```
sfs/                 library: data reading, model, inference, instance rule, metric
scripts/predict.py   frames -> submission.csv
scripts/evaluate_oof.py   out-of-fold maps -> PQ and the other reported numbers
demo.ipynb           the whole inference pipeline, step by step
weights/             the five trained models
artifacts/           out-of-fold maps, fold split, submitted file, rule search, training log, evaluation output
training/            the training script
analysis/            studies reported in the paper (see below)
report/              LaTeX source of the report and the script that draws its figures
```

## Analyses (`analysis/`)

| Script | Question | Answer |
|---|---|---|
| `annotator_ceiling.py` | How do human annotators score against each other? | PQ 0.34; a perfect copy of one annotator would get 0.76 |
| `error_autopsy.py` | Where is PQ lost, and what would a perfect fix of each step give? | see `artifacts/error_autopsy.json` |
| `selector_candidates.py`, `selector_expected_pq.py` | Can a learned, PQ-optimal region selector beat the fixed rule? | No: 0.428 vs 0.424 (negative result) |
| `average_maps.py` | Does adding a ResNeXt-50 ensemble help? | 0.429, not worth twice the cost (negative result) |

## Rules compliance

Only the competition's training images and annotations were used for training (plus ImageNet-pretrained encoder
weights from `timm`). No external labels, no test-set labels, no leaderboard-driven tuning.

## License

MIT (see `LICENSE`).
