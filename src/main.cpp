#include "Modulator.hpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

using namespace cuOFDM;
int main() {
  auto mod = Modulator<QAM256>();
  auto input = std::vector<uint8_t>(4096);

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{0, 255};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::generate(input.begin(), input.end(), gen);

  std::cout << input[0] << ' ' << input[3000] << std::endl;

  mod << input;
  mod.process();
  auto output = mod.getBatch();

  std::ofstream outfile;
  outfile.open("test.iq", std::ofstream::binary);
  std::cout << (*output)[0];
  outfile.write((const char *)&(*output).front(),
                input.size() * sizeof(std::complex<float>));

  return 0;
}
