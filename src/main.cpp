#include "Demodulator.hpp"
#include "Modulator.hpp"
#include "Streamer.hpp"
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>

using namespace cuOFDM;
int main() {
  auto pipeline = std::vector<std::shared_ptr<Pipe>>();
  auto mod = std::make_shared<Modulator<QAM256>>();
  auto demod = std::make_shared<Demodulator<QAM256>>();

  cudaStream_t aStream;
  cudaStreamCreate(&aStream);

  pipeline.push_back(mod);
  pipeline.push_back(demod);

  auto top = std::make_shared<Streamer>(pipeline, &aStream);
  (*top)();

  return 0;
}
