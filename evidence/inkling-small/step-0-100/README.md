# Inkling-Small training curves, steps 0–100

The CSV records four VERL metrics from one checkpoint lineage. Training steps
1–40 come from the first run, 41–80 from its continuation, and 81–100 from a
restart at checkpoint 80. The initial evaluation is step 0. The restart's
step-80 evaluation agrees with the original completed-step result.

The validation set is the same 30 AIME2025 questions at every evaluation.
Decoding uses temperature 1, sampling enabled, and one response per question;
the curve is noisy and the peak should not be treated as the final score.
Training reward uses varying prompt batches. The mismatch metric is mean
absolute **probability** difference over sampled response tokens, not
log-probability difference. A training-batch metric does not exist at step 0.

`plot.py` renders the four PNGs from `metrics.csv`. These are combined-stack
measurements, not an isolated ablation of a single PR.
