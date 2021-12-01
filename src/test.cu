#include "test.cuh"
#include <cstdio>

__global__ void bitsToIq(cuComplex *dMap, uint8_t *dIn, cuComplex *dOut) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dOut[tid] = dMap[dIn[tid]];
  printf("hello");
}

void callBitsToIq(cuComplex *dMap, uint8_t *dIn, cuComplex *dOut, size_t size) {
  bitsToIq<<<1, size>>>(dMap, dIn, dOut);
}

