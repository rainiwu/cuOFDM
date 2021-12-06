#ifndef Tester_h
#define Tester_h

#include "Demodulator.hpp"
#include "Modulator.hpp"
#include "Randomizer.hpp"
#include "Streamer.hpp"
#include <algorithm>
#include <array>
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

} // namespace Tester
} // namespace cuOFDM

#endif
