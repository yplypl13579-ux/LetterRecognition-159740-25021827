#include "backpropagation.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
}
std::string trim(std::string s) {
  auto b = s.find_first_not_of(" \r\n\t");
  if (b == s.npos)
    return "";
  return s.substr(b, s.find_last_not_of(" \r\n\t") - b + 1);
}
void validate(const NetworkConfig &c) {
  require(c.hidden.size() >= 2 && c.hidden.size() <= 5, "Use 2 to 5 hidden layers.");
  for (int n : c.hidden)
    require(n >= 2 && n <= 256, "Hidden layer sizes must be 2..256.");
  require(c.activation == "tanh" || c.activation == "relu", "Activation must be tanh or relu.");
  require(std::isfinite(c.learningRate) && c.learningRate > 0 && c.learningRate <= 1,
          "Learning rate must be in (0,1].");
  require(std::isfinite(c.momentum) && c.momentum >= 0 && c.momentum < 1,
          "Momentum must be in [0,1).");
  require(c.batchSize >= 1 && c.batchSize <= 4096, "Batch size must be 1..4096.");
}
} // namespace
LetterStructure parsePattern(const std::string &line, bool requireLabel) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string part;
  while (std::getline(ss, part, ','))
    fields.push_back(trim(part));
  require(!line.empty() && line.back() != ',', "Trailing comma is invalid.");
  bool labelled = fields.size() == 17;
  require(fields.size() == 16 || labelled,
          "Expected exactly 16 features, optionally preceded by A-Z.");
  require(!requireLabel || labelled, "Dataset rows require an A-Z label.");
  LetterStructure p;
  if (labelled) {
    require(fields[0].size() == 1 && fields[0][0] >= 'A' && fields[0][0] <= 'Z',
            "Label must be A-Z.");
    p.symbol = fields[0][0] - 'A';
  }
  for (int i = 0; i < 16; ++i) {
    size_t used = 0;
    const auto &v = fields[i + (labelled ? 1 : 0)];
    double x = std::stod(v, &used);
    require(used == v.size() && std::isfinite(x) && x >= 0 && x <= 15,
            "Features must be numeric and in [0,15].");
    p.f[i] = x;
  }
  return p;
}
Dataset readDataFile(const std::string &path) {
  std::ifstream f(std::filesystem::u8path(path));
  require(bool(f), "Cannot read dataset: " + path);
  Dataset data;
  std::string line;
  int n = 0;
  while (std::getline(f, line)) {
    ++n;
    if (trim(line).empty())
      continue;
    try {
      data.push_back(parsePattern(line, true));
    } catch (const std::exception &e) {
      throw std::runtime_error(path + " line " + std::to_string(n) + ": " + e.what());
    }
  }
  require(!data.empty(), "Dataset is empty.");
  return data;
}
Backpropagation::Backpropagation(NetworkConfig c) : cfg(c) {
  validate(cfg);
  initialise();
}
void Backpropagation::initialise() {
  validate(cfg);
  rng.seed(123);
  completedEpochs = 0;
  sizes = {16};
  sizes.insert(sizes.end(), cfg.hidden.begin(), cfg.hidden.end());
  sizes.push_back(26);
  weights.clear();
  velocity.clear();
  gradient.clear();
  a.clear();
  delta.clear();
  for (int n : sizes) {
    a.emplace_back(n, 0);
    delta.emplace_back(n, 0);
  }
  for (size_t l = 1; l < sizes.size(); ++l) {
    int fi = sizes[l - 1], fo = sizes[l];
    // He uniform for ReLU hidden layers; Xavier uniform for tanh and output.
    double limit = (cfg.activation == "relu" && l + 1 < sizes.size()) ? std::sqrt(6.0 / fi)
                                                                      : std::sqrt(6.0 / (fi + fo));
    Matrix w(fo, std::vector<double>(fi + 1, 0));
    for (auto &row : w)
      for (int i = 0; i < fi; ++i)
        row[i] = (2.0 * (double(rng()) / 4294967296.0) - 1.0) * limit;
    weights.push_back(w);
    velocity.emplace_back(fo, std::vector<double>(fi + 1, 0));
    gradient.emplace_back(fo, std::vector<double>(fi + 1, 0));
  }
}
void Backpropagation::setTrainingOptions(double lr, double m, int b, bool shuffle) {
  auto next = cfg;
  next.learningRate = lr;
  next.momentum = m;
  next.batchSize = b;
  next.shuffle = shuffle;
  validate(next);
  cfg = next;
}
void Backpropagation::forward(const LetterStructure &p) {
  for (int i = 0; i < 16; ++i)
    a[0][i] = p.f[i] / 15.0;
  for (size_t l = 0; l < weights.size(); ++l) {
    for (size_t o = 0; o < weights[l].size(); ++o) {
      const auto &row = weights[l][o];
      double z = row.back();
      for (size_t i = 0; i < a[l].size(); ++i)
        z += row[i] * a[l][i];
      a[l + 1][o] = z;
    }
    if (l + 1 < weights.size())
      for (double &z : a[l + 1])
        z = cfg.activation == "tanh" ? std::tanh(z) : std::max(0.0, z);
  }
  double mx = *std::max_element(a.back().begin(), a.back().end()), sum = 0;
  for (double &z : a.back()) {
    z = std::exp(z - mx);
    sum += z;
  }
  require(std::isfinite(sum) && sum > 0, "Non-finite network output: lower the learning rate.");
  for (double &z : a.back())
    z /= sum;
}
double Backpropagation::loss(int label) const {
  return -std::log(std::max(a.back()[label], 1e-15));
}
std::array<double, 26> Backpropagation::testNetwork(const LetterStructure &p) {
  forward(p);
  std::array<double, 26> out;
  std::copy(a.back().begin(), a.back().end(), out.begin());
  return out;
}
void Backpropagation::clearGradient() {
  for (auto &m : gradient)
    for (auto &row : m)
      std::fill(row.begin(), row.end(), 0.0);
}
void Backpropagation::backward(int label) {
  require(label >= 0 && label < 26, "Training requires a valid label.");
  // Softmax + cross entropy: dL/dz = probability - one-hot target.
  delta.back() = a.back();
  delta.back()[label] -= 1;
  for (int l = int(weights.size()) - 1; l > 0; --l) {
    std::fill(delta[l].begin(), delta[l].end(), 0.0);
    for (size_t o = 0; o < weights[l].size(); ++o)
      for (size_t i = 0; i < delta[l].size(); ++i)
        delta[l][i] += weights[l][o][i] * delta[l + 1][o];
    for (size_t i = 0; i < delta[l].size(); ++i)
      delta[l][i] *= cfg.activation == "tanh" ? 1 - a[l][i] * a[l][i] : (a[l][i] > 0 ? 1.0 : 0.0);
  }
  for (size_t l = 0; l < weights.size(); ++l)
    for (size_t o = 0; o < weights[l].size(); ++o) {
      for (size_t i = 0; i < a[l].size(); ++i)
        gradient[l][o][i] += delta[l + 1][o] * a[l][i];
      gradient[l][o].back() += delta[l + 1][o];
    }
}
void Backpropagation::trainEpoch(const Dataset &data) {
  require(!data.empty(), "Load training data first.");
  std::vector<size_t> order(data.size());
  std::iota(order.begin(), order.end(), 0);
  if (cfg.shuffle)
    for (size_t n = order.size(); n > 1; --n) {
      // Rejection sampling avoids modulo bias and library-specific shuffles.
      uint64_t bound = (uint64_t(1) << 32) / n * n;
      uint32_t r;
      do {
        r = rng();
      } while (r >= bound);
      std::swap(order[n - 1], order[r % n]);
    }
  for (size_t begin = 0; begin < data.size(); begin += cfg.batchSize) {
    clearGradient();
    size_t end = std::min(data.size(), begin + cfg.batchSize);
    for (size_t k = begin; k < end; ++k) {
      forward(data[order[k]]);
      backward(data[order[k]].symbol);
    }
    for (size_t l = 0; l < weights.size(); ++l)
      for (size_t o = 0; o < weights[l].size(); ++o)
        for (size_t i = 0; i < weights[l][o].size(); ++i) {
          velocity[l][o][i] = cfg.momentum * velocity[l][o][i] -
                              cfg.learningRate * gradient[l][o][i] / double(end - begin);
          weights[l][o][i] += velocity[l][o][i];
        }
  }
  ++completedEpochs;
}
NetworkMetrics Backpropagation::evaluate(const Dataset &data) {
  require(!data.empty(), "Cannot evaluate an empty dataset.");
  NetworkMetrics m;
  m.count = data.size();
  for (const auto &p : data) {
    require(p.symbol >= 0 && p.symbol < 26, "Evaluation requires labels.");
    forward(p);
    int predicted = int(std::max_element(a.back().begin(), a.back().end()) - a.back().begin());
    m.correct += predicted == p.symbol;
    ++m.confusion[p.symbol][predicted];
    m.mcel += loss(p.symbol);
    for (int j = 0; j < 26; ++j) {
      double e = a.back()[j] - (j == p.symbol ? 1.0 : 0.0);
      m.mse += e * e;
    }
  }
  m.mcel /= m.count;
  m.mse /= double(m.count * 26);
  m.pgc = 100.0 * m.correct / m.count;
  return m;
}
void Backpropagation::saveWeights(const std::string &path) const {
  std::ofstream out(std::filesystem::u8path(path));
  require(bool(out), "Cannot save weights: " + path);
  out << std::setprecision(17);
  out << "LETTER_NET_V2\n" << cfg.activation << ' ' << cfg.hidden.size();
  for (int n : cfg.hidden)
    out << ' ' << n;
  out << '\n'
      << cfg.learningRate << ' ' << cfg.momentum << ' ' << cfg.batchSize << ' ' << cfg.shuffle
      << ' ' << completedEpochs << "\n";
  for (const auto &m : weights)
    for (const auto &row : m) {
      for (double v : row)
        out << v << ' ';
      out << '\n';
    }
  for (const auto &m : velocity)
    for (const auto &row : m) {
      for (double v : row)
        out << v << ' ';
      out << '\n';
    }
  out << rng << "\nEND\n";
  out.flush();
  require(bool(out), "Writing weights failed.");
}
void Backpropagation::loadWeights(const std::string &path) {
  std::ifstream in(std::filesystem::u8path(path));
  require(bool(in), "Cannot read weights: " + path);
  std::string magic;
  in >> magic;
  require(magic == "LETTER_NET_V2", "Unsupported weight file. Use the V2 weights.");
  NetworkConfig c;
  size_t count = 0;
  in >> c.activation >> count;
  require(count >= 2 && count <= 5, "Invalid number of hidden layers.");
  c.hidden.resize(count);
  for (int &n : c.hidden)
    in >> n;
  int epochs = 0;
  in >> c.learningRate >> c.momentum >> c.batchSize >> c.shuffle >> epochs;
  require(bool(in) && epochs >= 0, "Invalid checkpoint configuration.");
  validate(c);
  Backpropagation candidate(c);
  for (auto &m : candidate.weights)
    for (auto &row : m)
      for (double &v : row) {
        in >> v;
        require(bool(in) && std::isfinite(v), "Invalid/truncated weight data.");
      }
  for (auto &m : candidate.velocity)
    for (auto &row : m)
      for (double &v : row) {
        in >> v;
        require(bool(in) && std::isfinite(v), "Invalid/truncated optimizer data.");
      }
  in >> candidate.rng >> magic;
  require(bool(in) && magic == "END", "Invalid RNG state or missing END.");
  std::string extra;
  require(!(in >> extra), "Unexpected trailing checkpoint content.");
  candidate.completedEpochs = epochs;
  *this = std::move(candidate);
}
double Backpropagation::gradientCheck() {
  auto original = *this;
  LetterStructure p;
  p.symbol = 4;
  for (int i = 0; i < 16; ++i)
    p.f[i] = (i * 7 + 3) % 16;
  forward(p);
  clearGradient();
  backward(p.symbol);
  auto analytic = gradient;
  double worst = 0, eps = 1e-6;
  for (size_t l = 0; l < weights.size(); ++l)
    for (size_t o = 0; o < weights[l].size(); ++o)
      for (size_t i = 0; i < weights[l][o].size(); ++i) {
        double old = weights[l][o][i];
        weights[l][o][i] = old + eps;
        forward(p);
        double plus = loss(p.symbol);
        weights[l][o][i] = old - eps;
        forward(p);
        double minus = loss(p.symbol);
        weights[l][o][i] = old;
        worst = std::max(worst, std::abs((plus - minus) / (2 * eps) - analytic[l][o][i]));
      }
  *this = original;
  return worst;
}
void Backpropagation::zeroForTest() {
  for (auto &m : weights)
    for (auto &row : m)
      std::fill(row.begin(), row.end(), 0.0);
}
void writeConfusion(const std::string &path, const NetworkMetrics &m) {
  std::ofstream out(std::filesystem::u8path(path));
  require(bool(out), "Cannot save confusion matrix.");
  out << "Actual/Predicted";
  for (int j = 0; j < 26; ++j)
    out << ',' << char('A' + j);
  out << '\n';
  for (int i = 0; i < 26; ++i) {
    out << char('A' + i);
    for (int n : m.confusion[i])
      out << ',' << n;
    out << '\n';
  }
  require(bool(out), "Confusion matrix write failed.");
}
void writeLogHeader(std::ostream &o) {
  o << "epoch,training_mcel,training_mse,training_pgc,validation_mcel,validation_mse,validation_"
       "pgc,test_mcel,test_mse,test_pgc\n";
}
void writeLogRow(std::ostream &o, int e, const NetworkMetrics &t, const NetworkMetrics &v,
                 const NetworkMetrics &s) {
  o << std::setprecision(17) << e << ',' << t.mcel << ',' << t.mse << ',' << t.pgc << ',' << v.mcel
    << ',' << v.mse << ',' << v.pgc << ',' << s.mcel << ',' << s.mse << ',' << s.pgc << '\n';
}
