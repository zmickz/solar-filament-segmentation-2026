# Solar filament segmentation — native-resolution U-Net ensemble with a metric-tuned instance rule

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(GONG H-alpha frames, MAGFiLO annotations).

| | |
|---|---|
| Out-of-fold Panoptic Quality (707 training frames) | **0.429** (95% bootstrap interval 0.420–0.437) |
| Public leaderboard | **0.38** |
| Model | 2 families × 5 U-Nets: ResNet-34 encoder (24 M parameters, 49 MB per file) and ResNeXt-50 32x4d encoder (32 M, 64 MB) |
| Training | 2 h 22 min (ResNet-34) + 3 h 11 min (ResNeXt-50) on 2 × Tesla T4, all folds including predictions |
| Inference | ResNet-34 family alone: about 2.6 s per frame on one T4; the full ensemble on CPU: about 4 min per frame |

## How it works

1. **Pre-processing.** Each 2048 × 2048 frame is standardised with the mean and standard deviation of the
   solar disk. No resizing.
2. **Network.** Two families of five U-Nets (one per date-grouped fold; ResNet-34 and ResNeXt-50 encoders),
   trained on 640-pixel crops with the same script. The target of a pixel is
   the *fraction of annotators* that marked it; the loss is binary cross-entropy + Dice.
3. **Inference.** Full frame in one pass, four flips, mean of the five models of each family, then mean of
   the two families.
4. **Instances.** 8-connected regions of `p > 0.25` (disjoint by construction). A region is kept when its area is
   at least 320 px and its mean probability at least 0.83. These numbers were chosen by grid search on
   out-of-fold ensemble maps, scored with the competition's pooled Panoptic Quality
   (`artifacts/rule_search.txt`; areas 260 and 320 tie, we kept 320).

The technical report is in [`report/`](report/).

## Reproduce the submission (no training)

```bash
pip install -r requirements.txt
python scripts/predict.py --images <MAGFiLO>/test/test_images --out submission.csv
```

or open [`demo.ipynb`](demo.ipynb), which runs the same pipeline step by step, compares the result with the
file we submitted (`artifacts/submission_kaggle.csv`) and shows a few segmentations.
The weights are in `weights/` and are loaded by both. A GPU is recommended (CPU: about four minutes per frame;
use `--limit 3` or `SFS_LIMIT=3` for a quick look).

## Reproduce the validation numbers

```bash
python scripts/evaluate_oof.py --annotations <MAGFiLO>/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
```

scores `artifacts/oof_maps.zip` (every training frame predicted by the fold models that never saw it, mean of
the two families) and
prints the pooled PQ, its bootstrap interval, per-fold values, IoU/Dice distribution and fragmentation counts.
Runs in about a minute on CPU. `--maps artifacts/oof_maps_resnet34.zip --min_mean 0.85` (or `oof_maps_resnext50.zip`)
scores each family alone (0.424 and 0.422).

## Retrain

`training/train_kaggle.py` is the exact script that produced the weights. It is a single file meant for a Kaggle
notebook with the competition data attached and two T4 GPUs: it builds a cache, rehearses every stage on a few
frames, trains the five folds, predicts, tunes the rule and writes the submission. As is, it trains the ResNet-34
family (40 minutes per fold); with `SFS_ENCODER=resnext50_32x4d SFS_BUDGET_S=3300` it trains the ResNeXt-50 family
(55 minutes per fold). `artifacts/training_log_kaggle_resnet34.log` and `artifacts/training_log_kaggle_resnext50.log`
are the logs of the two runs and `artifacts/folds.json` the split (shared). The two families' test maps were averaged
with `analysis/average_maps.py` and the rule re-tuned with `analysis/rule_grid.py`.

## Repository layout

```
sfs/                 library: data reading, model, inference, instance rule, metric
scripts/predict.py   frames -> submission.csv
scripts/evaluate_oof.py   out-of-fold maps -> PQ and the other reported numbers
demo.ipynb           the whole inference pipeline, step by step
weights/             the ten trained models (2 encoders x 5 folds)
artifacts/           out-of-fold maps, fold split, submitted files, rule searches, training logs, evaluation outputs
                     (files without a suffix belong to the final ensemble; *_resnet34 to the first submission)
training/            the training script
analysis/            studies reported in the paper (see below)
report/              LaTeX source of the report and the script that draws its figures
```

## Analyses (`analysis/`)

| Script | Question | Answer |
|---|---|---|
| `annotator_ceiling.py` | How do human annotators score against each other? | PQ 0.34; a perfect copy of one annotator would get 0.76 |
| `error_autopsy.py` | Where is PQ lost, and what would a perfect fix of each step give? | see `artifacts/error_autopsy.json` |
| `selector_candidates.py`, `selector_expected_pq.py` | Can a learned, PQ-optimal region selector beat the fixed rule? | No: 0.428 vs 0.424 on the ResNet-34 maps (negative result) |
| `average_maps.py`, `rule_grid.py` | Does adding a ResNeXt-50 family help? | Yes, a little: 0.424 → 0.429 out of fold, 0.37 → 0.38 public; this is the final solution |

## Rules compliance

Only the competition's training images and annotations were used for training (plus ImageNet-pretrained encoder
weights from `timm`). No external labels, no test-set labels. Every choice was made on out-of-fold scores; the public
leaderboard was only used to confirm them (one stricter rule, mean ≥ 0.89, lost 0.008 out of fold and 0.01 public,
and was not adopted).

## License

MIT (see `LICENSE`).
