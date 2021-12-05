#ifndef Modulator_cuh
#define Modulator_cuh

#include "Common.hpp"
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

/** Modulates input buffer, saves to output buffer, expects size of BATCH_SIZE
  * inBuff, outBuff are device pointers
  */
void process(uint8_t* inBuff, cuComplex* outBuff, cuComplex* dMap, cudaStream_t* aStream); 

#endif
