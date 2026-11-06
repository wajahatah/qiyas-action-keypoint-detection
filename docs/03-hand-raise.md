# 3 · Hand raise — a negative result, documented

**Goal:** detect a candidate raising a hand to call the invigilator. It looks like the easiest of
the three behaviours. Six approaches in five weeks (24 Nov – 26 Dec 2024) showed it is not, and
mostly taught me how evaluation can mislead.

## Research log

| Dates | Step | Result |
|---|---|---|
| 24–26 Nov | 9 annotated videos; shallow classifier on **all persons' keypoints** at once | baseline |
| 26–28 Nov | larger set; compared all-person input vs **one row per person** | per-person input adopted |
| 28 Nov – 5 Dec | 30 annotated videos; normalisation study: divide by frame size vs subtract seat-ROI origin | frame-size normalisation kept |
| 3–10 Dec | keypoint MLPs (single- and two-branch x/y) and Random Forest; first test-bench inference | not satisfactory on unseen video |
| 11–16 Dec | systematic test of 9 models on the bench; RF tuned with k-fold + grid search | best honest result below |
| 18–19 Dec | image approaches: YOLO classifier on seat crops; YOLOv10s detector on frames with keypoints drawn | high validation scores, failed on test |
| 22–26 Dec | hybrid: EfficientNet-B0 image features + keypoint MLP (multi-person and single-crop variants) | no stable improvement |

Dataset: 25 videos → 37,307 person-frames, 22.5 % hand-raise; per-video positive rate ranged from
1.4 % to 53 %.

## Results

| Approach | Evaluation | Result |
|---|---|---|
| Elbow-angle + wrist-above-shoulder rule | qualitative only | starting point; no bench numbers recorded |
| **Random Forest on normalised keypoints** | 5 held-out videos (7,811 person-frames) | **hand-raise P 0.80 · R 0.43 · F1 0.56** |
| same RF | 1 unseen test video | no detection on any frame |
| YOLO classifier on seat crops | frame-level val split, 3 % positives | **97.3 % accuracy = the all-negative baseline**; recall 0–11 % |
| YOLOv10s detector, keypoints overlaid | frame-level val split | **mAP50 0.995** — leakage |
| same detector | 1 unseen test video | 3 detections in 160 frames |
| Hybrid EfficientNet-B0 + keypoint MLP | — | no reliable gain; training logs not preserved |

Source: [`results/hand_raise_results.csv`](../results/hand_raise_results.csv). Code:
[`src/qiyas_akd/hand_raise/`](../src/qiyas_akd/hand_raise/).

The live system used a separate detector for this behaviour, and hand raise was removed from the
live alert set in February 2025.

## What went wrong, precisely

1. **Accuracy paradox.** With 3 % positives, a classifier that always says "no" scores 97 %. The
   crop classifier's 97.3 % was exactly that.
2. **Frame-level leakage.** The detector's validation frames were consecutive with its training
   frames. 0.995 mAP measured memorisation; the unseen video exposed it.
3. **Silent data bugs** found while re-auditing the RF:
   - one video's label column header had a trailing space (`"hand_raised "`), so 858 rows became NaN
     labels → a phantom third class in the report;
   - the grid search scored with `recall` on that 3-class target → every candidate scored NaN, so the
     "best" parameters were simply the first in the grid;
   - one normalisation script divided y by 780 instead of 720;
   - the hybrid model's accuracy summed over 4 outputs but divided by batch size, so it could exceed 1.

   The re-implementation validates column names and labels on load and fixes each of these.
4. **Probably too few independent events.** Hand raises are short and rare, and positives were very
   unevenly spread (1.4 %–53 % per video); 25 videos are likely too few independent events to learn
   from keypoints alone.

These lessons fed directly into hand in pocket ([§4](04-hand-in-pocket.md)): event-level test
benches from day one, explicit label rules, and balancing by segment rather than by row.
