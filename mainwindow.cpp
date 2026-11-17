#include "mainwindow.h"
#include "default_config.h"
#include <QtWidgets>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>

// The starter MainWindow workflows (load, initialize, train, classify, evaluate,
// save/load weights) are retained; fixed arrays and the .ui layout are replaced.
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), bp(bestDefaults()) {
  setWindowTitle("Letter Recognition - Peilin Yang 25021827");
  resize(1140, 860);
  auto central = new QWidget;
  setCentralWidget(central);
  auto layout = new QVBoxLayout(central);
  auto title = new QLabel("Letter Recognition | Multi-layer neural network");
  title->setStyleSheet("font-size:20px;font-weight:600;color:#183b56;padding:8px");
  layout->addWidget(title);
  dataPanel = new QWidget;
  auto dataLayout = new QVBoxLayout(dataPanel);
  dataLayout->setContentsMargins(0, 0, 0, 0);
  auto pathLayout = new QHBoxLayout;
  directoryEdit = new QLineEdit(QDir::currentPath());
  auto browse = new QPushButton("Choose data folder");
  auto read = new QPushButton("Read all three CSV files");
  pathLayout->addWidget(directoryEdit, 1);
  pathLayout->addWidget(browse);
  pathLayout->addWidget(read);
  dataLayout->addLayout(pathLayout);
  auto fileButtons = new QHBoxLayout;
  const QStringList names = {"Load training CSV...", "Load validation CSV...", "Load test CSV..."};
  for (int i = 0; i < 3; ++i) {
    auto button = new QPushButton(names[i]);
    fileButtons->addWidget(button);
    connect(button, &QPushButton::clicked, this, [this, i] {
      auto f = QFileDialog::getOpenFileName(this, "Select labelled CSV", directoryEdit->text(),
                                            "CSV files (*.csv);;All files (*)");
      if (!f.isEmpty())
        guarded([&] { loadSplit(i, f); });
    });
  }
  dataLayout->addLayout(fileButtons);
  layout->addWidget(dataPanel);
  statusLabel = new QLabel;
  layout->addWidget(statusLabel);
  connect(browse, &QPushButton::clicked, this, [this] {
    auto dir = QFileDialog::getExistingDirectory(this, "Data folder", directoryEdit->text());
    if (!dir.isEmpty())
      directoryEdit->setText(dir);
  });
  connect(read, &QPushButton::clicked, this,
          [this] { guarded([&] { loadAll(directoryEdit->text()); }); });
  settingsPanel = new QGroupBox("Network and training settings");
  auto form = new QGridLayout(settingsPanel);
  hiddenEdit = new QLineEdit;
  activationBox = new QComboBox;
  activationBox->addItems({"tanh", "relu"});
  learningBox = new QDoubleSpinBox;
  learningBox->setDecimals(4);
  learningBox->setRange(0.0001, 1);
  learningBox->setSingleStep(.01);
  momentumBox = new QDoubleSpinBox;
  momentumBox->setDecimals(2);
  momentumBox->setRange(0, .99);
  momentumBox->setSingleStep(.05);
  batchBox = new QSpinBox;
  batchBox->setRange(1, 4096);
  epochsBox = new QSpinBox;
  epochsBox->setRange(1, 10000);
  epochsBox->setValue(DEFAULT_MAX_EPOCHS);
  targetBox = new QDoubleSpinBox;
  targetBox->setRange(0, 100);
  targetBox->setDecimals(2);
  targetBox->setValue(100);
  targetBox->setSuffix(" %");
  shuffleBox = new QCheckBox("Shuffle before each epoch (seed 123)");
  form->addWidget(new QLabel("Hidden nodes (comma separated)"), 0, 0);
  form->addWidget(hiddenEdit, 0, 1);
  form->addWidget(new QLabel("Hidden activation"), 0, 2);
  form->addWidget(activationBox, 0, 3);
  form->addWidget(new QLabel("Learning rate"), 1, 0);
  form->addWidget(learningBox, 1, 1);
  form->addWidget(new QLabel("Momentum"), 1, 2);
  form->addWidget(momentumBox, 1, 3);
  form->addWidget(new QLabel("Batch size"), 2, 0);
  form->addWidget(batchBox, 2, 1);
  form->addWidget(new QLabel("Maximum additional epochs"), 2, 2);
  form->addWidget(epochsBox, 2, 3);
  form->addWidget(new QLabel("Stop at training PGC"), 3, 0);
  form->addWidget(targetBox, 3, 1);
  form->addWidget(shuffleBox, 3, 2, 1, 2);
  layout->addWidget(settingsPanel);
  actionsPanel = new QWidget;
  auto actions = new QHBoxLayout(actionsPanel);
  actions->setContentsMargins(0, 0, 0, 0);
  auto init = new QPushButton("Initialize (reset weights)");
  auto load = new QPushButton("Load weights...");
  auto save = new QPushButton("Save current weights...");
  auto shuffle = new QPushButton("Shuffle training data now");
  for (auto b : {init, load, save, shuffle}) {
    actions->addWidget(b);
  }
  layout->addWidget(actionsPanel);
  connect(init, &QPushButton::clicked, this, [this] { guarded([&] { initialiseNetwork(); }); });
  connect(load, &QPushButton::clicked, this, [this] {
    auto f = QFileDialog::getOpenFileName(this, "Load V2 model", directoryEdit->text(),
                                          "Weights (*.txt)");
    if (!f.isEmpty())
      guarded([&] { loadModel(f); });
  });
  connect(save, &QPushButton::clicked, this, [this] {
    auto f = QFileDialog::getSaveFileName(
        this, "Save model and optimizer state",
        QDir(directoryEdit->text()).filePath("current_weights.txt"), "Weights (*.txt)");
    if (!f.isEmpty())
      guarded([&] {
        bp.saveWeights(f.toStdString());
        results->appendPlainText("Weights saved: " + f);
      });
  });
  connect(shuffle, &QPushButton::clicked, this, [this] {
    guarded([&] {
      if (training.empty())
        throw std::runtime_error("Load training data first.");
      std::mt19937 r(123);
      std::shuffle(training.begin(), training.end(), r);
      results->appendPlainText("Training rows shuffled in memory (seed 123). Manual row reordering "
                               "changes the subsequent training trajectory.");
    });
  });
  auto runRow = new QHBoxLayout;
  trainButton = new QPushButton("Train / continue training");
  stopButton = new QPushButton("Stop after current epoch");
  stopButton->setEnabled(false);
  runRow->addWidget(trainButton);
  runRow->addWidget(stopButton);
  for (int i = 0; i < 3; ++i) {
    auto b = new QPushButton(
        QString("Evaluate %1").arg(QStringList{"training", "validation", "test"}[i]));
    runRow->addWidget(b);
    connect(b, &QPushButton::clicked, this, [this, i] {
      if (!running)
        guarded([&] { evaluateSplit(i); });
    });
  }
  layout->addLayout(runRow);
  connect(trainButton, &QPushButton::clicked, this, [this] { guarded([&] { startTraining(); }); });
  connect(stopButton, &QPushButton::clicked, this, [this] { stopRequested = true; });
  patternPanel = new QGroupBox("Classify one pattern (16 features in [0,15]; optional A-Z label)");
  auto patternLayout = new QVBoxLayout(patternPanel);
  auto inputRow = new QHBoxLayout;
  patternEdit = new QLineEdit("2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8");
  auto predict = new QPushButton("Classify");
  inputRow->addWidget(patternEdit);
  inputRow->addWidget(predict);
  patternLayout->addLayout(inputRow);
  classificationLabel = new QLabel("Load best_weights.txt to use the submitted classifier.");
  patternLayout->addWidget(classificationLabel);
  outputTable = new QTableWidget(2, 13);
  outputTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
  outputTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  outputTable->horizontalHeader()->hide();
  outputTable->verticalHeader()->hide();
  outputTable->setMaximumHeight(104);
  patternLayout->addWidget(outputTable);
  layout->addWidget(patternPanel);
  connect(predict, &QPushButton::clicked, this, [this] { guarded([&] { classify(); }); });
  results = new QPlainTextEdit;
  results->setReadOnly(true);
  results->setMinimumHeight(170);
  layout->addWidget(results, 1);
  syncControls();
  refreshStatus();
  if (QFileInfo::exists(QDir::current().filePath("training_set.csv"))) {
    guarded([&] {
      loadAll(QDir::currentPath());
      if (QFileInfo::exists("best_weights.txt"))
        loadModel(QDir::current().filePath("best_weights.txt"));
    });
  }
}
void MainWindow::guarded(const std::function<void()> &action) {
  try {
    action();
  } catch (const std::exception &e) {
    results->appendPlainText("ERROR: " + QString::fromUtf8(e.what()));
  }
}
void MainWindow::refreshStatus() {
  statusLabel->setText(
      QString("Loaded rows - Training: %1 | Validation: %2 | Test: %3 | Model epoch: %4")
          .arg(training.size())
          .arg(validation.size())
          .arg(test.size())
          .arg(bp.epochs()));
}
void MainWindow::loadAll(const QString &dir) {
  QDir d(dir);
  auto tr = readDataFile(d.filePath("training_set.csv").toStdString());
  auto va = readDataFile(d.filePath("validation_set.csv").toStdString());
  auto te = readDataFile(d.filePath("test_set.csv").toStdString());
  training = std::move(tr);
  validation = std::move(va);
  test = std::move(te);
  dataDirectory = d.absolutePath();
  directoryEdit->setText(dataDirectory);
  refreshStatus();
  results->appendPlainText("Loaded training_set.csv, validation_set.csv and test_set.csv from " +
                           dataDirectory);
}
void MainWindow::loadSplit(int i, const QString &path) {
  auto data = readDataFile(path.toStdString());
  if (i == 0)
    training = std::move(data);
  else if (i == 1)
    validation = std::move(data);
  else
    test = std::move(data);
  refreshStatus();
  results->appendPlainText("Loaded: " + path);
}
void MainWindow::syncControls() {
  const auto &c = bp.config();
  QStringList sizes;
  for (int n : c.hidden)
    sizes << QString::number(n);
  hiddenEdit->setText(sizes.join(','));
  activationBox->setCurrentText(QString::fromStdString(c.activation));
  learningBox->setValue(c.learningRate);
  momentumBox->setValue(c.momentum);
  batchBox->setValue(c.batchSize);
  shuffleBox->setChecked(c.shuffle);
}
void MainWindow::initialiseNetwork() {
  NetworkConfig c;
  c.hidden.clear();
  for (const auto &part : hiddenEdit->text().split(',')) {
    bool ok = false;
    int n = part.trimmed().toInt(&ok);
    if (!ok)
      throw std::runtime_error("Hidden nodes must be comma-separated integers.");
    c.hidden.push_back(n);
  }
  c.activation = activationBox->currentText().toStdString();
  c.learningRate = learningBox->value();
  c.momentum = momentumBox->value();
  c.batchSize = batchBox->value();
  c.shuffle = shuffleBox->isChecked();
  bp = Backpropagation(c);
  refreshStatus();
  results->appendPlainText("Network initialized with seed 123. Previous weights were reset.");
}
void MainWindow::loadModel(const QString &path) {
  bp.loadWeights(path.toStdString());
  syncControls();
  refreshStatus();
  results->appendPlainText("Loaded weights, momentum and RNG state: " + path);
}
void MainWindow::classify() {
  auto p = parsePattern(patternEdit->text().toStdString(), false);
  auto outputs = bp.testNetwork(p);
  int predicted = int(std::max_element(outputs.begin(), outputs.end()) - outputs.begin());
  classificationLabel->setText(
      QString("Predicted: %1 | Probability: %2% | Sum: %3%4")
          .arg(QChar('A' + predicted))
          .arg(outputs[predicted] * 100, 0, 'f', 3)
          .arg(std::accumulate(outputs.begin(), outputs.end(), 0.0), 0, 'f', 8)
          .arg(p.symbol < 0 ? QString()
                            : QString(" | Expected: %1 (%2)")
                                  .arg(QChar('A' + p.symbol))
                                  .arg(p.symbol == predicted ? "correct" : "incorrect")));
  for (int i = 0; i < 26; ++i) {
    outputTable->setItem(
        i / 13, i % 13,
        new QTableWidgetItem(QString("%1\n%2").arg(QChar('A' + i)).arg(outputs[i], 0, 'e', 3)));
    outputTable->item(i / 13, i % 13)->setToolTip(QString::number(outputs[i], 'g', 17));
  }
  outputTable->resizeRowsToContents();
}
void MainWindow::showMetrics(const QString &name, const NetworkMetrics &m) {
  results->appendPlainText(QString("%1: n=%2, correct=%3, MCEL=%4, MSE=%5, PGC=%6%")
                               .arg(name)
                               .arg(m.count)
                               .arg(m.correct)
                               .arg(m.mcel, 0, 'f', 8)
                               .arg(m.mse, 0, 'f', 8)
                               .arg(m.pgc, 0, 'f', 4));
}
void MainWindow::evaluateSplit(int i) {
  showMetrics(QStringList{"Training", "Validation", "Test"}[i], bp.evaluate(i == 0   ? training
                                                                            : i == 1 ? validation
                                                                                     : test));
  if (i == 2) {
    auto path = QFileDialog::getSaveFileName(
        this, "Export test confusion matrix (optional)",
        QDir(directoryEdit->text()).filePath("current_confusion_matrix.csv"), "CSV (*.csv)");
    if (!path.isEmpty()) {
      writeConfusion(path.toStdString(), bp.evaluate(test));
      results->appendPlainText("Confusion matrix saved: " + path);
    }
  }
}
void MainWindow::setBusy(bool busy) {
  running = busy;
  for (auto w : {settingsPanel, dataPanel, actionsPanel, patternPanel})
    w->setEnabled(!busy);
  trainButton->setEnabled(!busy);
  stopButton->setEnabled(busy);
}
void MainWindow::startTraining() {
  if (training.empty() || validation.empty() || test.empty())
    throw std::runtime_error("Load all three datasets before training.");
  QStringList nodes;
  for (int n : bp.config().hidden)
    nodes << QString::number(n);
  if (nodes.join(',') != hiddenEdit->text().remove(' ') ||
      activationBox->currentText().toStdString() != bp.config().activation)
    throw std::runtime_error(
        "Click Initialize to apply changed architecture/activation, or load a model.");
  bp.setTrainingOptions(learningBox->value(), momentumBox->value(), batchBox->value(),
                        shuffleBox->isChecked());
  runTarget = targetBox->value();
  auto t = bp.evaluate(training);
  if (t.pgc >= runTarget) {
    showMetrics("Training", t);
    results->appendPlainText("Target PGC already reached. No updates performed.");
    return;
  }
  runDirectory =
      QDir(directoryEdit->text())
          .filePath("gui_runs/" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss_zzz"));
  if (!QDir().mkpath(runDirectory))
    throw std::runtime_error("Cannot create run folder.");
  log = std::make_unique<std::ofstream>(
      std::filesystem::u8path(QDir(runDirectory).filePath("training_log.txt").toStdString()));
  if (!*log)
    throw std::runtime_error("Cannot create training log.");
  writeLogHeader(*log);
  auto v = bp.evaluate(validation);
  writeLogRow(*log, bp.epochs(), t, v, bp.evaluate(test));
  bestValidation = v.mcel;
  bp.saveWeights(QDir(runDirectory).filePath("best_weights.txt").toStdString());
  runRemaining = epochsBox->value();
  stopRequested = false;
  setBusy(true);
  results->appendPlainText("Training started. Results: " + runDirectory);
  QTimer::singleShot(0, this, [this] { trainNextEpoch(); });
}
void MainWindow::trainNextEpoch() {
  try {
    if (stopRequested) {
      finishTraining("Stopped by user.");
      return;
    }
    bp.trainEpoch(training);
    auto t = bp.evaluate(training), v = bp.evaluate(validation), s = bp.evaluate(test);
    writeLogRow(*log, bp.epochs(), t, v, s);
    log->flush();
    if (!*log)
      throw std::runtime_error("Log write failed.");
    if (v.mcel < bestValidation) {
      bestValidation = v.mcel;
      bp.saveWeights(QDir(runDirectory).filePath("best_weights.txt").toStdString());
    }
    results->appendPlainText(QString("Epoch %1 | Training MCEL %2 PGC %3% | Validation MCEL %4 PGC "
                                     "%5% | Test MCEL %6 PGC %7%")
                                 .arg(bp.epochs())
                                 .arg(t.mcel, 0, 'f', 5)
                                 .arg(t.pgc, 0, 'f', 2)
                                 .arg(v.mcel, 0, 'f', 5)
                                 .arg(v.pgc, 0, 'f', 2)
                                 .arg(s.mcel, 0, 'f', 5)
                                 .arg(s.pgc, 0, 'f', 2));
    refreshStatus();
    if (t.pgc >= runTarget) {
      finishTraining("Target training PGC reached.");
      return;
    }
    if (--runRemaining <= 0) {
      finishTraining("Maximum additional epochs reached.");
      return;
    }
    QTimer::singleShot(0, this, [this] { trainNextEpoch(); });
  } catch (const std::exception &e) {
    log.reset();
    setBusy(false);
    results->appendPlainText("ERROR: " + QString::fromUtf8(e.what()));
  }
}
void MainWindow::finishTraining(const QString &reason) {
  log.reset();
  setBusy(false);
  bp.saveWeights(QDir(runDirectory).filePath("last_weights.txt").toStdString());
  results->appendPlainText(reason + " Current model saved as last_weights.txt. Best validation "
                                    "checkpoint saved separately as best_weights.txt.");
}
bool MainWindow::smokeTest(const QString &scratch) {
  auto require = [](bool ok, const char *msg) {
    if (!ok)
      throw std::runtime_error(msg);
  };
  require(training.size() == 16000 && validation.size() == 3200 && test.size() == 4000,
          "GUI dataset counts wrong");
  require(training.front().f == readDataFile("training_set.csv").front().f &&
              test.front().f == readDataFile("test_set.csv").front().f &&
              validation.front().f == readDataFile("validation_set.csv").front().f,
          "GUI split loading mismatch");
  require(bp.epochs() > 0, "GUI did not load checkpoint");
  auto original = bp;
  showMetrics("Training", bp.evaluate(training));
  showMetrics("Validation", bp.evaluate(validation));
  showMetrics("Test", bp.evaluate(test));
  classify();
  require(outputTable->item(1, 12) != nullptr, "26 outputs not displayed");
  bool failed = false;
  try {
    loadSplit(2, QDir(scratch).filePath("missing.csv"));
  } catch (...) {
    failed = true;
  }
  require(failed && test.size() == 4000, "Failed dataset load changed test set");
  targetBox->setValue(0);
  int epoch = bp.epochs();
  startTraining();
  require(!running && epoch == bp.epochs(), "Already-reached target failed");
  // One actual epoch through the same GUI training handler; output stays in scratch.
  directoryEdit->setText(scratch);
  epochsBox->setValue(1);
  targetBox->setValue(100);
  startTraining();
  while (running)
    QCoreApplication::processEvents();
  require(bp.epochs() == epoch + 1, "GUI maximum epoch stop failed");
  bp = original;
  directoryEdit->setText(dataDirectory);
  epochsBox->setValue(DEFAULT_MAX_EPOCHS);
  targetBox->setValue(100);
  syncControls();
  refreshStatus();
  classify();
  results->appendPlainText("GUI SMOKE TEST PASSED: correct datasets, 26 outputs, failed-load "
                           "preservation, target stop, one-epoch training and max-epoch stop.");
  return true;
}
