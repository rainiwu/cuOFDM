#ifndef Tester_h
#define Tester_h

#include "Demodulator.hpp"
#include "Modulator.hpp"
#include "Randomizer.hpp"
#include "Streamer.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace cuOFDM {
namespace Tester {

int modDemod();

int pipedModDemod();

int modRandDemod();

int throughput();

} // namespace Tester
} // namespace cuOFDM

#endif
