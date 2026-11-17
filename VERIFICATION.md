# V3 verification record

Peilin Yang - 25021827. Verification performed on 7 September 2026. This V3 record supersedes the V2 record; V3 trains the final network on the 16,000-row training set required by the assignment PDF.

## Protocol change in V3 and how it was produced

- `training_set.csv` now contains 16,000 rows: the v5 starter's 12,800-row training file followed, in original order, by its 3,200-row validation file. The first 12,800 rows are byte-identical to the starter training file; the last 3,200 rows are byte-identical to the starter validation file. `validation_set.csv` and `test_set.csv` are unchanged.
- `experiments.cpp` implements a two-stage protocol. Stage 1 trains six candidates on the first 12,800 rows and selects by validation MCEL (run 5, ReLU 16-64-32-26, learning rate 0.03, epoch 143). Stage 2 retrains that configuration from scratch on all 16,000 rows for exactly 143 epochs. Test rows are evaluated every epoch but never select or stop anything.
- Stage 1 was re-run under the new layout and produced log files byte-identical to the V2 stage-1 logs for all six runs, proving the 12,800-row slice reproduces the original split exactly.
- The GUI smoke-test dataset check was updated from 12,800 to 16,000 training rows. The GUI trains on all 16,000 rows.

## Build and execution

- Installed toolchain: Qt 6.11.2, MinGW GCC 13.1.0 64-bit, Qt Creator 20.0.1. Build and tests ran from PowerShell/Git Bash on Windows x64, build 26200.
- `experiments.exe` (g++ -O2 -Wall -Wextra, backpropagation.cpp + experiments.cpp) compiled with no warnings. `--self-test` passed: tanh gradient error 3.64708e-10, ReLU 3.9495e-10, plus parsing, softmax, determinism, loss reduction, exact resume, atomic load and metric checks.
- The two-stage run completed all six 150-epoch stage-1 experiments and the 143-epoch stage-2 final run.

## Engine tests

The submitted `experiments.cpp --self-test` checks:

- Valid and invalid pattern parsing, including wrong feature counts and non-finite inputs.
- Tanh and ReLU finite-difference gradients. Maximum absolute errors were 3.64708e-10 and 3.9495e-10 respectively, below the 1e-5 tolerance.
- Finite softmax outputs and sum one, deterministic initialization, and synthetic training loss reduction.
- Equal inference before and after saving/loading.
- Exact resumed training: after one additional epoch, the loaded and uninterrupted models produced identical complete checkpoint text, including optimizer and RNG state.
- Invalid/truncated checkpoint loading leaves the previous model intact.
- Uniform-output MCEL ln(26), MSE 25/676, and confusion count consistency.

## Independent numerical and data checks

Two independent implementations recomputed the final model's metrics from the packaged `best_weights.txt`:

1. A separately written C++ verifier (loads the checkpoint, evaluates the three CSV files, compares the confusion matrix against `confusion_matrix.csv`): 0 mismatches.
2. A NumPy verifier that parses `best_weights.txt` directly (weights, biases, softmax) without using the C++ engine. All metrics matched the C++ values within 1e-12, and the 26 x 26 test confusion matrix matched exactly.

| Split | Examples | Correct | PGC |
| --- | ---: | ---: | ---: |
| Training | 16,000 | 15,446 | 96.5375% |
| Validation file (in-sample for the final model) | 3,200 | 3,092 | 96.625% |
| Test | 4,000 | 3,725 | 93.125% |

The submitted checkpoint is the stage-2 final model: ReLU 16-64-32-26, epoch 143 of 143, test MCEL 0.22114756, MSE 0.00386734, 3,725/4,000 correct. Stage 1 selected the configuration and epoch count using validation MCEL only; test metrics never selected anything.

`data_manifest.json` records the row counts and SHA-256 hashes of all three CSV files and documents the derivation of the merged training set from the supplied v5 archive.

## Workbook and PDF checks

- `performance_and_confusion.xlsx` was rebuilt for V3: stage-1 six-run comparison with validation-rank formulas, final-model learning curves (144 rows, epochs 0-143, training and test MCEL/MSE/PGC with three native editable charts), and the 26 x 26 test confusion matrix with A-Z headers, live Total/Recall/Correct/PGC formulas. All confusion values were programmatically matched against `confusion_matrix.csv`.
- `bestNetwork_25021827.xlsx` values updated to the final model's metrics; notes document the two-stage protocol and that validation-file rows are in-sample for the final model.
- REPORT.pdf (five pages) regenerated: results table, 16,000-row data protocol, full derivations, two-stage experiment discussion (including the honest comparison with the stage-1 checkpoint's 93.800% test PGC), user guide, verification and references.
- `checklist.xlsx` is unchanged from V2 (its answers remain accurate: all deliverables still present and 'full').

## Archive checks

The submission package contains source, build descriptions, data, model state, actual experiment logs, spreadsheets and documentation. Executables, object files, generated Makefiles, temporary tests, preview images, scratch directories and inspection sidecars are excluded from the ZIP. The final ZIP is checked against the packaged source files for identical bytes.

## Items requiring student/instructor attention

- The V3 protocol follows the assignment PDF's 16,000 training rows. The v5 starter's own three-way split (12,800/3,200/4,000) is preserved inside the two-stage procedure: stage 1 reproduces the starter's validation-based selection, stage 2 trains on all 16,000 rows. If the instructor instead intends the v5 files to be used as-is (12,800 gradient-training rows), the V2 package remains available.
- The stage-2 fixed epoch count (143) means the final model is not validation-checkpointed; its 93.125% test PGC is 0.675 points below the stage-1 validation-selected checkpoint. This trade-off is disclosed in REPORT.pdf.
- AI generated substantial implementation and documentation. Review `AI_USAGE_RECORD.md` and confirm whether that scope is permitted. The checklist's planning/debugging language may be narrower.
- Confirm the complete title and instructions of the second submission portal; the supplied screenshot is cropped and contains a conflicting Assignment #2 line.
- Review and understand the implementation before submitting. This record does not certify independent student authorship or student review.
