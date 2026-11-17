#pragma once
#include "backpropagation.h"
inline NetworkConfig bestDefaults() {
  NetworkConfig c;
  c.hidden = {64, 32};
  c.activation = "relu";
  c.learningRate = 0.03;
  return c;
}
constexpr int DEFAULT_MAX_EPOCHS = 150;
