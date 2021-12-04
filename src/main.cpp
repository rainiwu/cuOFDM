#include "Demodulator.hpp"
#include "Modulator.hpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

using namespace cuOFDM;
int main() {
  auto mod = Modulator<QAM256>();
  auto input = std::vector<uint8_t>(BATCH_SIZE * 30);

  auto demod_input = std::vector<std::complex<float>>(BATCH_SIZE * 30);

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, 255};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::generate(input.begin(), input.end(), gen);

  std::cout << input[0] << ' ' << input[3000] << std::endl;
  auto copy = input;

  std::ofstream outfile;
  outfile.open("test.iq", std::ofstream::binary | std::ios_base::app);
  for (int i = 0; i < 30; i++) {
    mod << input;
    input.erase(input.begin(), input.begin() + BATCH_SIZE);
    mod.cpuProcess();
    auto output = mod.getBatch();

    for (int j = 0; j < BATCH_SIZE; j++) {
      demod_input[j + BATCH_SIZE * i] = (*output)[j];
    }

    std::cout << (*output)[0] << std::endl;
    outfile.write((const char *)&(*output).front(),
                  BATCH_SIZE * sizeof(std::complex<float>));
  }

  auto demod = Demodulator<QAM256>();
  for (int i = 0; i < 30; i++) {
    demod << demod_input;
    demod_input.erase(demod_input.begin(), demod_input.begin() + BATCH_SIZE);
    demod.cpuDemod();
    auto output = demod.getBatch();

    auto result = std::string("PASS");
    for (size_t j = 0; j < BATCH_SIZE; j++) {
      if ((*output)[j] != copy[j + i * BATCH_SIZE]) {
        result = std::string("FAIL");
        std::cout << (int)(*output)[j] << " " << (int)copy[j + i * BATCH_SIZE]
                  << std::endl;
        break;
      }
    }
    std::cout << result << std::endl;
  }

  return 0;
}
