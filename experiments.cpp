#include "backpropagation.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
namespace fs = std::filesystem;
void check(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
}
std::string readText(const fs::path &p) {
  std::ifstream f(p);
  return std::string(std::istreambuf_iterator<char>(f), {});
}
void tests(const fs::path &scratch) {
  fs::create_directories(scratch);
  auto p = parsePattern("A,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8", true);
  check(p.symbol == 0 && p.f[6] == 13, "CSV parsing failed");
  check(parsePattern("2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8", false).symbol == -1,
        "Unlabelled parsing failed");
  for (const std::string bad :
       {"A,1,2", "?,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8", "A,nan,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8",
        "A,20,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8"}) {
    bool rejected = false;
    try {
      parsePattern(bad, true);
    } catch (...) {
      rejected = true;
    }
    check(rejected, "Malformed pattern accepted");
  }
  for (const std::string act : {"tanh", "relu"}) {
    NetworkConfig c;
    c.hidden = {5, 4};
    c.activation = act;
    Backpropagation net(c);
    double error = net.gradientCheck();
    std::cout << act << " gradient max absolute error: " << error << '\n';
    check(error < 1e-5, "Gradient check failed");
  }
  NetworkConfig c;
  c.hidden = {8, 6};
  c.batchSize = 7;
  Backpropagation a(c), b(c);
  auto pa = a.testNetwork(p), pb = b.testNetwork(p);
  check(pa == pb, "Seed reproducibility failed");
  check(std::abs(std::accumulate(pa.begin(), pa.end(), 0.0) - 1) < 1e-12, "Softmax sum failed");
  Dataset data;
  for (int n = 0; n < 31; ++n) {
    auto q = p;
    q.symbol = n % 2;
    for (int j = 0; j < 16; ++j)
      q.f[j] = (n % 2) ? 12 : 2;
    data.push_back(q);
  }
  double before = a.evaluate(data).mcel;
  for (int e = 0; e < 10; ++e)
    a.trainEpoch(data);
  check(a.evaluate(data).mcel < before, "Training did not reduce loss");
  a.saveWeights((scratch / "resume.txt").u8string());
  b.loadWeights((scratch / "resume.txt").u8string());
  check(a.testNetwork(p) == b.testNetwork(p), "Reload changed inference");
  a.trainEpoch(data);
  b.trainEpoch(data);
  a.saveWeights((scratch / "a.txt").u8string());
  b.saveWeights((scratch / "b.txt").u8string());
  check(readText(scratch / "a.txt") == readText(scratch / "b.txt"), "Resume not bitwise identical");
  std::ofstream(scratch / "bad.txt") << "LETTER_NET_V2\ntanh 2 8 6\n0.03 0.85 7 1 10\n";
  auto old = b.testNetwork(p);
  bool failed = false;
  try {
    b.loadWeights((scratch / "bad.txt").u8string());
  } catch (...) {
    failed = true;
  }
  check(failed && old == b.testNetwork(p), "Failed load changed model");
  b.zeroForTest();
  auto uniform = b.evaluate(data);
  check(std::abs(uniform.mcel - std::log(26)) < 1e-12, "Uniform cross entropy incorrect");
  check(std::abs(uniform.mse - 25.0 / 676) < 1e-12, "MSE definition incorrect");
  int total = 0, trace = 0;
  for (int i = 0; i < 26; ++i) {
    trace += uniform.confusion[i][i];
    for (int v : uniform.confusion[i])
      total += v;
  }
  check(total == 31 && trace == int(uniform.correct), "Confusion totals incorrect");
  std::cout << "PASS: parsing, gradient, softmax, deterministic initialization, loss reduction, "
               "exact resume, atomic load, metrics, confusion\n";
}
int main(int argc, char **argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--self-test") {
      tests(argc > 2 ? fs::u8path(argv[2]) : fs::path("test_scratch"));
      return 0;
    }
    if (argc < 4) {
      std::cerr << "Usage: experiments DATA_DIR OUTPUT_DIR SCRATCH_DIR [epochs=150]\n       "
                   "experiments --self-test SCRATCH_DIR\n\n"
                   "Two-stage protocol (assignment PDF: 16,000 training rows, 4,000 test rows):\n"
                   "  Stage 1 - selection: six candidates train on the first 12,800 rows; the\n"
                   "             3,200-row validation file selects checkpoints, the winning\n"
                   "             configuration and the epoch count. Test rows never select.\n"
                   "  Stage 2 - final: the winning configuration is retrained from scratch on\n"
                   "             all 16,000 training rows for the selected epoch count. The\n"
                   "             submitted weights, logs, metrics and confusion matrix come\n"
                   "             from this final 16,000-row run.\n";
      return 2;
    }
    fs::path dataDir = fs::u8path(argv[1]), outDir = fs::u8path(argv[2]),
             scratch = fs::u8path(argv[3]);
    int epochs = argc > 4 ? std::stoi(argv[4]) : 150;
    check(epochs > 0, "Epochs must be positive");
    fs::create_directories(outDir / "experiments");
    fs::create_directories(scratch);
    Dataset full = readDataFile((dataDir / "training_set.csv").u8string()),
            va = readDataFile((dataDir / "validation_set.csv").u8string()),
            te = readDataFile((dataDir / "test_set.csv").u8string());
    check(full.size() == 16000 && va.size() == 3200 && te.size() == 4000,
          "Expected assignment split: 16000 training / 3200 validation-file / 4000 test rows");
    // Stage 1 trains on the first 12,800 rows (the supplied v5 training rows, in order), so
    // the validation file stays held out while the configuration is being selected.
    Dataset tr(full.begin(), full.begin() + 12800);
    std::ofstream summary(outDir / "experiment_summary.csv");
    summary << "run,activation,hidden1,hidden2,learning_rate,momentum,batch,seed,max_epochs,best_"
               "epoch,training_mcel,training_mse,training_pgc,validation_mcel,validation_mse,"
               "validation_pgc,test_mcel,test_mse,test_pgc\n"
            << std::setprecision(17);
    std::vector<NetworkConfig> candidates;
    for (const std::string act : {"tanh", "relu"}) {
      for (auto hidden : {std::vector<int>{32, 16}, std::vector<int>{64, 32}}) {
        NetworkConfig c;
        c.hidden = hidden;
        c.activation = act;
        candidates.push_back(c);
      }
      NetworkConfig c;
      c.hidden = {64, 32};
      c.activation = act;
      c.learningRate = 0.01;
      candidates.push_back(c);
    }
    double bestLoss = 1e300;
    int selectedRun = -1;
    Backpropagation selected;
    for (size_t r = 0; r < candidates.size(); ++r) {
      auto cfg = candidates[r];
      Backpropagation net(cfg), best(cfg);
      double minLoss = 1e300;
      std::string id = "run_" + std::to_string(r + 1);
      std::ofstream log(outDir / "experiments" / (id + ".csv"));
      writeLogHeader(log);
      writeLogRow(log, 0, net.evaluate(tr), net.evaluate(va), net.evaluate(te));
      for (int e = 1; e <= epochs; ++e) {
        net.trainEpoch(tr);
        auto tm = net.evaluate(tr), vm = net.evaluate(va), sm = net.evaluate(te);
        writeLogRow(log, e, tm, vm, sm);
        if (vm.mcel < minLoss) {
          minLoss = vm.mcel;
          best = net;
        } // Test metrics NEVER used to select.
        if (e % 25 == 0)
          std::cout << id << " " << cfg.activation << " epoch=" << e << " train=" << tm.pgc
                    << " validation=" << vm.pgc << " val_MCEL=" << vm.mcel << std::endl;
      }
      auto tm = best.evaluate(tr), vm = best.evaluate(va), sm = best.evaluate(te);
      summary << id << ',' << cfg.activation << ',' << cfg.hidden[0] << ',' << cfg.hidden[1] << ','
              << cfg.learningRate << ',' << cfg.momentum << ',' << cfg.batchSize << ",123,"
              << epochs << ',' << best.epochs() << ',' << tm.mcel << ',' << tm.mse << ',' << tm.pgc
              << ',' << vm.mcel << ',' << vm.mse << ',' << vm.pgc << ',' << sm.mcel << ',' << sm.mse
              << ',' << sm.pgc << '\n';
      summary.flush();
      if (vm.mcel < bestLoss) {
        bestLoss = vm.mcel;
        selected = best;
        selectedRun = int(r) + 1;
      }
    }
    // ---- Stage 2: final run on all 16,000 training rows -----------------------
    // The winning configuration and the epoch count were fixed by the validation-based
    // selection above. The final network is retrained from scratch on the full 16,000-row
    // training set required by the assignment PDF; the 4,000 test rows are evaluated
    // every epoch for the required curves but never influence training or stopping.
    NetworkConfig winner = selected.config();
    int finalEpochs = selected.epochs();
    Backpropagation finalNet(winner);
    std::ofstream finalLog(outDir / "final_training_log.csv");
    finalLog << "epoch,training_mcel,training_mse,training_pgc,test_mcel,test_mse,test_pgc\n"
             << std::setprecision(17);
    double lastTestMcel = 0;
    for (int e = 0; e <= finalEpochs; ++e) {
      if (e > 0)
        finalNet.trainEpoch(full);
      auto tm = finalNet.evaluate(full), sm = finalNet.evaluate(te);
      lastTestMcel = sm.mcel;
      finalLog << e << ',' << tm.mcel << ',' << tm.mse << ',' << tm.pgc << ',' << sm.mcel << ','
               << sm.mse << ',' << sm.pgc << '\n';
      finalLog.flush();
      if (e % 25 == 0 || e == finalEpochs)
        std::cout << "final(16000) epoch=" << e << " train_pgc=" << tm.pgc
                  << " test_pgc=" << sm.pgc << " test_MCEL=" << sm.mcel << std::endl;
    }
    finalLog.close();
    finalNet.saveWeights((outDir / "best_weights.txt").u8string());
    Backpropagation reloaded;
    reloaded.loadWeights((outDir / "best_weights.txt").u8string());
    check(reloaded.epochs() == finalEpochs && reloaded.config().activation == winner.activation,
          "Reloaded final model differs");
    auto tm = reloaded.evaluate(full), vm = reloaded.evaluate(va), sm = reloaded.evaluate(te);
    check(std::abs(sm.mcel - lastTestMcel) < 1e-13, "Reloaded final model metrics differ");
    writeConfusion((outDir / "confusion_matrix.csv").u8string(), sm);
    fs::copy_file(outDir / "final_training_log.csv", outDir / "training_log.txt",
                  fs::copy_options::overwrite_existing);
    std::ofstream info(outDir / "best_result.txt");
    info << std::setprecision(17) << "protocol=two_stage_16000_training\n"
         << "selection_run=" << selectedRun
         << "\nselection_activation=" << winner.activation
         << "\nselection_hidden1=" << winner.hidden[0]
         << "\nselection_hidden2=" << winner.hidden[1]
         << "\nselection_learning_rate=" << winner.learningRate
         << "\nselection_epoch=" << finalEpochs << "\nselection_validation_mcel=" << bestLoss
         << "\nfinal_training_rows=" << full.size() << "\nfinal_epochs=" << finalEpochs
         << "\nactivation=" << winner.activation << "\nhidden1=" << winner.hidden[0]
         << "\nhidden2=" << winner.hidden[1] << "\nlearning_rate=" << winner.learningRate
         << "\nepoch=" << finalEpochs << "\nmax_epochs=" << epochs
         << "\ntraining_mcel=" << tm.mcel << "\ntraining_mse=" << tm.mse
         << "\ntraining_pgc=" << tm.pgc << "\nvalidation_mcel=" << vm.mcel
         << "\nvalidation_mse=" << vm.mse << "\nvalidation_pgc=" << vm.pgc
         << "\ntest_mcel=" << sm.mcel << "\ntest_mse=" << sm.mse << "\ntest_pgc=" << sm.pgc
         << "\ntest_correct=" << sm.correct << "\n";
    std::cout << "STAGE1 selected run " << selectedRun << " (" << winner.activation << " "
              << winner.hidden[0] << "-" << winner.hidden[1] << ", lr=" << winner.learningRate
              << ") at epoch " << finalEpochs << ", validation MCEL=" << bestLoss << std::endl;
    std::cout << "STAGE2 final 16000-row model: TEST PGC=" << sm.pgc << " MCEL=" << sm.mcel
              << " correct=" << sm.correct << "/" << te.size() << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
