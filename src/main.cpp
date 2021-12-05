#include "Demodulator.hpp"
#include "Modulator.hpp"
#include "Streamer.hpp"
#include <algorithm>
#include <array>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>

using namespace cuOFDM;
int main() {
  auto in = std::array<uint8_t, BATCH_SIZE>();

  std::random_device aDev;
  std::mt19937 me{aDev()};
  std::uniform_int_distribution<int> dist{0, 255};
  auto gen = [&dist, &me]() { return dist(me); };
  std::generate(in.begin(), in.end(), gen);

  auto pipeline = std::vector<std::shared_ptr<Pipe>>();
  auto mod = std::make_shared<Modulator<QAM256>>();
  auto demod = std::make_shared<Demodulator<QAM256>>();

  cudaStream_t aStream;
  cudaStreamCreate(&aStream);

  pipeline.push_back(mod);
  pipeline.push_back(demod);

  auto top = std::make_shared<Streamer>(pipeline, &aStream);
  (*top) << in;

  (*top)();
  cudaStreamSynchronize(aStream);

  auto out = std::array<uint8_t, BATCH_SIZE>();
  (*top) >> out;

  auto result = std::string("PASS");
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    if (in[i] != out[i]) {
      result = std::string("FAIL");
      break;
    }
  }

  std::cout << result << std::endl;

  return 0;
}
