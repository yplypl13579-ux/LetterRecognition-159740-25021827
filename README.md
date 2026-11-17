# Assignment 1 - Neural Network Engine and Character Recognition

Peilin Yang - 25021827 - 159.740 Intelligent Systems

This V3 submission supersedes the earlier drafts. It trains the final network on the full 16,000-row training set required by the assignment PDF. Use only V3 source, V3 weights and V3 results together.

## Start in Qt Creator

1. Extract the submission ZIP to a writable folder.
2. Open `LetterRecognition.pro` in Qt Creator. Select the Desktop Qt 6.11.2 MinGW 64-bit kit.
3. In Projects > Run, set Working directory to the extracted project folder containing the three CSV files and `best_weights.txt`.
4. Build and Run. When files are in the working directory, the application automatically loads all three datasets and the best checkpoint. Otherwise use Choose data folder, Read all three CSV files, then Load weights.
5. Click Evaluate training, Evaluate validation, or Evaluate test. Expected final-model test output: 4,000 rows, 3,725 correct, PGC 93.1250%, MCEL 0.22114756, MSE 0.00386734. The test button offers optional CSV confusion-matrix export.

Qt Creator 20.0.1 and Qt 6.11.2 / MinGW GCC 13.1.0 were available for this build. The actual compile and GUI smoke tests used that installed Qt/MinGW toolchain from the command line. No interactive Qt Creator build is claimed.

## Build without the IDE

From a Qt/MinGW command prompt in this folder, run:

```text
mingw32-make
mingw32-make test
```

The wrapper Makefile creates `build/` and runs qmake there. It does not overwrite itself. `build.ps1` is an alternative PowerShell helper that finds Qt and MinGW under `C:\Qt`; run `./build.ps1` if your PowerShell execution policy permits local scripts. Do not change security policy to run it; use Qt Creator or the Qt command prompt instead. If several incompatible Qt/MinGW versions are installed, select the matching kit in Qt Creator.

## Classify one pattern

Enter exactly 16 comma-separated features in [0,15], for example:

```text
2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8
```

An optional uppercase label followed by a comma may precede the features. Click Classify. The application shows the predicted class, confidence, probability sum and all 26 numeric outputs. A supplied label additionally enables a correct/incorrect comparison. Malformed input is rejected with a message in the results panel.

## Train, stop and resume

- Hidden nodes accepts 2 to 5 comma-separated layer sizes, each 2..256. Activation accepts tanh or relu. Click Initialize to apply architecture changes; this resets weights, momentum, epoch and RNG to seed 123.
- Learning rate, momentum, batch size and automatic shuffling apply at the next training call. Defaults match run 5: ReLU, hidden layers 64 and 32, learning rate 0.03, momentum 0.85, batch 64, shuffling enabled, 150 maximum additional epochs and target training PGC 100%.
- Click Train / continue training. The GUI trains on all 16,000 rows of `training_set.csv`. It stops after the requested additional epochs, on meeting the training-PGC target, or after Stop after current epoch is clicked. If the target is already met, it makes no updates.
- Every completed epoch logs training, validation and test MCEL, MSE and PGC to a new `gui_runs/<timestamp>/training_log.txt`. Evaluation takes place after updates, on a fixed model. Epoch 0 in experiment logs is the initialization baseline. In this V3 protocol the validation rows are part of the 16,000-row training file, so the logged validation curve is in-sample for a GUI run; it is informative only.
- Each GUI run writes its lowest-validation-MCEL checkpoint to `best_weights.txt` in that run folder, and its current ending state to `last_weights.txt`. The supplied best weights in the project root are not automatically overwritten.
- Load weights restores architecture, hyperparameters, completed epochs, momentum buffers and RNG state. Then click Train to continue. Do not click Initialize after loading if you want to continue.
- Exact continuation requires the same dataset in the same order and unchanged training options. Shuffle training data now explicitly reorders the in-memory dataset; using it changes the subsequent trajectory relative to the recorded experiments. Automatic per-epoch shuffling operates on the example traversal order.
- Dataset loading is transactional. If a file is invalid, the previously loaded split remains intact. Missing files and invalid weights produce an ERROR message, not a success message.

## Reproduce all experiments

Build the command-line runner, then execute from the project folder:

```text
mingw32-make experiments
build\experiments.exe . reproduced_results build\experiment_scratch 150
```

This runner and the GUI compile the same `backpropagation.cpp`. They use the same normalization, initialization, RNG, mini-batch updates and checkpoint format. The runner implements the two-stage protocol:

- Stage 1 (selection): six candidate configurations train on the first 12,800 rows of `training_set.csv` (the v5 starter's original training rows). The 3,200-row `validation_set.csv` selects each run's best checkpoint and the winning configuration and epoch count, using minimum validation MCEL only. Test metrics are logged every epoch but never used for selection. Ties retain the earlier epoch/run.
- Stage 2 (final training): the winning configuration (ReLU 16-64-32-26, learning rate 0.03) is retrained from scratch on all 16,000 training rows for the selected 143 epochs. The submitted `best_weights.txt`, `final_training_log.csv`, `confusion_matrix.csv` and `best_result.txt` come from this final run.

For GUI reproduction of the final model: load the CSVs, set default parameters, click Initialize, and train 143 epochs. The stage-1 winner can be reproduced by loading only the first 12,800 training rows through the training-file chooser.

## Data protocol

The assignment PDF specifies 16,000 training rows and 4,000 test rows. This submission follows that requirement:

- `training_set.csv` contains 16,000 rows: the v5 starter's 12,800-row training file followed, in original order, by its 3,200-row validation file. The first 12,800 rows are byte-identical to the starter's training file, and the data comes from the supplied v5 archive with no other changes.
- `validation_set.csv` (3,200 rows) is retained unchanged. It is held out only during stage-1 configuration selection, where stage 1 trains on the first 12,800 rows.
- `test_set.csv` (4,000 rows) is retained byte-identical and is never used for any selection decision.
- The final submitted network receives gradient updates from all 16,000 training rows.

`data_manifest.json` records the row counts and SHA-256 hashes of all three files and documents the derivation.

## Submission files

- `checklist.xlsx`: original template, only requested yellow cells filled; student details are on row 5.
- `bestNetwork_25021827.xlsx`: filled network specification in the supplied template.
- `performance_and_confusion.xlsx`: six-run comparison, editable MCEL/MSE/PGC charts and labelled 26x26 test confusion matrix.
- `REPORT.pdf`: architecture, derivatives, experiments, discussion, short guide, verification and references.
- `AI_USAGE_RECORD.md`: prompts, response excerpts and substantive AI contribution disclosure. It is a selected record, not a verbatim export of the whole conversation.
- `best_weights.txt`, `training_log.txt` (= `final_training_log.csv`), `final_training_log.csv`, `experiment_summary.csv`, `best_result.txt`, `confusion_matrix.csv`, `experiments/*.csv`: actual outputs of the shared C++ engine.
- `backpropagation.*`, `mainwindow.*`, `main.cpp`, `default_config.h`, `experiments.cpp`, `LetterRecognition.pro`, `Makefile`, `build.ps1`: complete source and build support. The Qt layout is built in C++, so a `.ui` file is not required.
- `VERIFICATION.md` and `data_manifest.json`: verification summary and source data hashes.

The two submission portals are intended for the standalone checklist and the complete assignment package respectively. Confirm the second portal's complete title/instructions before uploading, since the supplied screenshot crops its title and includes a conflicting 'Assignment #2' line.

Only source, documentation, data and results belong in the submission archive. Build outputs and test scratch files are excluded.
