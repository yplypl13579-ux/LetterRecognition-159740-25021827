#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

// Replaces the supplied starter Backpropagation engine. Shared by Qt and CLI.
struct LetterStructure {
  std::array<double, 16> f{};
  int symbol = -1;
};
using Dataset = std::vector<LetterStructure>;
Dataset readDataFile(const std::string &path);
LetterStructure parsePattern(const std::string &line, bool requireLabel);
struct NetworkConfig {
  std::vector<int> hidden{64, 32};
  std::string activation = "tanh";
  double learningRate = 0.03, momentum = 0.85;
  int batchSize = 64;
  bool shuffle = true;
};
struct NetworkMetrics {
  double mcel = 0, mse = 0, pgc = 0;
  size_t count = 0, correct = 0;
  std::array<std::array<int, 26>, 26> confusion{};
};
class Backpropagation {
public:
  explicit Backpropagation(NetworkConfig config = {});
  void initialise();
  std::array<double, 26> testNetwork(const LetterStructure &p);
  void trainEpoch(const Dataset &data);
  NetworkMetrics evaluate(const Dataset &data);
  void saveWeights(const std::string &path) const;
  void loadWeights(const std::string &path); // atomic validation, includes resume state
  const NetworkConfig &config() const { return cfg; }
  int epochs() const { return completedEpochs; }
  void setTrainingOptions(double lr, double momentum, int batch, bool shuffle);
  double gradientCheck();
  void zeroForTest();

private:
  using Matrix = std::vector<std::vector<double>>;
  NetworkConfig cfg;
  std::vector<int> sizes;
  std::vector<Matrix> weights, velocity, gradient;
  std::vector<std::vector<double>> a, delta;
  std::mt19937 rng{123};
  int completedEpochs = 0;
  void forward(const LetterStructure &p);
  void backward(int label);
  void clearGradient();
  double loss(int label) const;
};
void writeConfusion(const std::string &path, const NetworkMetrics &m);
void writeLogHeader(std::ostream &out);
void writeLogRow(std::ostream &out, int epoch, const NetworkMetrics &tr, const NetworkMetrics &va,
                 const NetworkMetrics &te);
