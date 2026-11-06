"""Hand-in-pocket recognition from temporal windows of the 10-keypoint skeleton.

This is the longest research line of the project (12 approaches, from single-frame random forests
to 3D-CNN hybrids). The approach that won on the video test benches — and was deployed — is a small
MLP over a 5-frame window of which only frames t0, t2, t4 are kept (3 x 20 keypoint values + 4 desk
position values = 64 inputs), trained on segment-balanced, jitter-augmented, normalised data.

Sub-packages:

* ``dataprep`` — windowing + labelling rules, normalisation, balancing, augmentation, and the
  config-driven pipeline that chains them (shared by training and inference);
* ``models`` — trees, MLP architectures c0..c4, LSTM/GRU, temporal CNNs, crop+keypoint hybrids;
* ``train`` — YAML-driven training (``python -m qiyas_akd.hand_in_pocket.train``);
* ``infer`` — streaming per-desk predictor and per-frame / event-level evaluation.
"""
