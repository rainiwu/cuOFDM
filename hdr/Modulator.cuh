#ifndef Modulator_cuh
#define Modulator_cuh

#include "Common.hpp"
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>

void processBitBatch(uint8_t* hBitBatch, uint8_t* dBitBatch, cuComplex* hModBatch, cuComplex* dModBatch, cuComplex* dMap);

#endif
