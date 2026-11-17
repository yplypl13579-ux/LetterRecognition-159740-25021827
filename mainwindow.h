#pragma once
#include "backpropagation.h"
#include <QMainWindow>
#include <QString>
#include <fstream>
#include <functional>
#include <memory>
class QLineEdit;
class QComboBox;
class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;
class QPushButton;
class QPlainTextEdit;
class QTableWidget;
class QLabel;
class QWidget;
class MainWindow : public QMainWindow {
public:
  explicit MainWindow(QWidget *parent = nullptr);
  bool smokeTest(const QString &scratch);

private:
  Backpropagation bp;
  Dataset training, validation, test;
  QString dataDirectory;
  QLineEdit *directoryEdit, *hiddenEdit, *patternEdit;
  QComboBox *activationBox;
  QDoubleSpinBox *learningBox, *momentumBox, *targetBox;
  QSpinBox *batchBox, *epochsBox;
  QCheckBox *shuffleBox;
  QPushButton *trainButton, *stopButton;
  QWidget *settingsPanel, *dataPanel, *actionsPanel, *patternPanel;
  QPlainTextEdit *results;
  QTableWidget *outputTable;
  QLabel *statusLabel, *classificationLabel;
  bool running = false, stopRequested = false;
  int runRemaining = 0;
  double runTarget = 100, bestValidation = 1e300;
  QString runDirectory;
  std::unique_ptr<std::ofstream> log;
  void guarded(const std::function<void()> &action);
  void loadAll(const QString &directory);
  void loadSplit(int split, const QString &path);
  void refreshStatus();
  void syncControls();
  void initialiseNetwork();
  void loadModel(const QString &path);
  void classify();
  void evaluateSplit(int split);
  void startTraining();
  void trainNextEpoch();
  void finishTraining(const QString &reason);
  void showMetrics(const QString &name, const NetworkMetrics &m);
  void setBusy(bool busy);
};
