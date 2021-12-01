#ifndef test_cuh
#define test_cuh

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstdint>

void callBitsToIq(cuComplex *dMap, uint8_t *dIn, cuComplex *dOut, size_t size);

#endif
