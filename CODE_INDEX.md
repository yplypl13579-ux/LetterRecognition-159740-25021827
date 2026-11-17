# Source-code index

Peilin Yang - 25021827 - 159.740 Assignment 1

This directory contains the complete C++ / Qt source code. Start with these files:

1. `main.cpp` - program startup and Qt application configuration.
2. `mainwindow.h` and `mainwindow.cpp` - the graphical user interface, file loading, controls, classification, training and evaluation actions.
3. `backpropagation.h` and `backpropagation.cpp` - shared neural-network engine: parsing, normalization, forward propagation, stable softmax, backpropagation, mini-batch momentum, metrics, checkpoint save/load and confusion matrix writing.
4. `experiments.cpp` - command-line experiment runner and automated engine tests.
5. `LetterRecognition.pro` - Qt/qmake project definition.
6. `Makefile` and `build.ps1` - build commands.

## Build and test

Open `LetterRecognition.pro` in Qt Creator and select the Desktop Qt 6.11.2 MinGW 64-bit kit. Alternatively, from a Qt/MinGW command prompt in this directory, run:

```text
mingw32-make
mingw32-make test
```

The application expects `training_set.csv`, `validation_set.csv`, `test_set.csv`, and `best_weights.txt` in its working directory. The complete submission package contains these runtime files. The source-only package is intentionally kept small so the code can be reviewed separately.

## Verification result

The source was compiled successfully with Qt 6.11.2 and MinGW GCC 13.1.0. Automated tests passed, including finite-difference gradient checks, deterministic initialization, checkpoint resume equivalence and metric/confusion-matrix checks. See `VERIFICATION.md` in the full submission for the detailed record.
