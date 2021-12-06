#ifndef Randomizer_cuh
#define Randomizer_cuh

#include <curand.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include "Common.hpp"

void initRand(curandState *aState);
void applyRand(cuComplex* inBuff, cuComplex* outBuff, curandState *aState, cudaStream_t *aStream);

#endif
