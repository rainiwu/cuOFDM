#ifndef Demodulator_cuh
#define Demodulator_cuh

#include "Common.hpp"
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>

void demod(cuComplex* inBuff, uint8_t *outBuff, cuComplex* dMap, cudaStream_t* aStream);

#endif
