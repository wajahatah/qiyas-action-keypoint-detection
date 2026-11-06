# 6 · Evaluation lessons

The most transferable thing this project taught me is how easily a number can look good and mean
nothing. Every result in this repository is labelled with its evaluation type
(`eval_type` column in [`results/`](../results/)):

| `eval_type` | What it is | How much to trust it |
|---|---|---|
| `video_bench_events` | fixed set of held-out videos, alert events counted as TP / FP / FN | **primary** — measures what invigilators experience |
| `live_alert_review_precision_only` | every alert raised in the live system reviewed by hand | real-world precision; **no recall** (missed events never logged) |
| `val_split_video_level` | validation split with whole videos held out | fair, but per-frame |
| `val_split_frame_level` | random frame-level split | **inflated by leakage** — shown only as training history |
| `test_video_qualitative` | one unseen video, inspected by eye | a smoke test |

## 1. Frame-level splits leak

Consecutive frames of a video are near-identical. Randomly splitting frames puts near-copies of
training images into validation.

- Pose model: validation mAP50-95 jumped from 0.80 to 0.99 when the dataset and split changed.
- Hand-raise detector: **mAP50 0.995** on validation, **3 detections in 160 frames** on an unseen
  video.

Fix in this repo: every split helper groups by video (`evaluation.grouped_split`,
`pose.dataset`).

## 2. Accuracy hides imbalance

The hand-raise crop classifier reached 97.3 % accuracy on a validation set with 3 % positives —
precisely the score of always answering "no". Report per-class precision / recall, or event-level
counts.

## 3. Freeze the ground truth

In the first hand-in-pocket evaluation round, the number of ground-truth events (TP + FN) drifted
between 29 and 70 across model rows, because events were re-judged per run. The second round fixed
the reference set (42 events) and the comparison became trustworthy. `evaluation.match_events`
makes event matching mechanical.

## 4. Precision-only logs are still useful — if you say so

Live review measured what mattered operationally (how many alerts were wrong) but not what was
missed. When the hand-in-pocket MLP cut false alerts by 92 %, its true alerts also fell; without FN
logging that recall cost can only be inferred, not measured. Next time: periodically sample
un-alerted footage to estimate recall in the field.

## 5. Silent bugs I found by re-auditing

| Bug | Effect | Guard in this repo |
|---|---|---|
| CSV header `"hand_raised "` (trailing space) in one file | 858 labels became NaN → phantom third class | column names stripped and labels validated on load |
| grid search scored with `recall` on that 3-class target | every candidate scored NaN; "best" = first in grid | binary target asserted before tuning |
| y divided by 780 instead of 720 | features slightly off-scale | one normalisation function, tested |
| accuracy summed over 4 outputs, divided by batch size | accuracy > 1 possible | per-output metrics |
| `flip_idx: []` in the pose dataset config | flip augmentation silently disabled | correct left/right swap table, tested |
| `optimizer=auto` with hand-set `lr0` | learning-rate sweeps had no effect | training wrapper warns on the combination |
| person → desk by head x-range only | a person whose head was not detected (masked to x = 0) fell into the leftmost desk | 2-D ROI test on the head, shoulder-midpoint fallback |
| throughput logged as 1 / batch latency incl. drawing | "FPS" was batches/s and included display time | benchmark times inference only, reports frames/s and batches/s |

## 6. Match training to deployment

The live system processes every 2nd frame. A model trained on consecutive frames saw a different
input distribution than it got in production; emulating the skip in the training windows (`t3`)
removed that mismatch. More generally: write down the deployment contract first
([§5](05-system-integration.md)) and make the training data obey it.
