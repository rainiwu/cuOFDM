#include "Modulator.cuh"
#include <cufft.h>

__global__ void modulate(uint8_t* dBits, cuComplex* dMods, cuComplex* dMap) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dMods[tid] = dMap[dBits[tid]];

}

void process(uint8_t* inBuff, cuComplex* outBuff, cuComplex* myMap, cufftHandle* myPlan, cudaStream_t* aStream) {
  modulate<<<BATCH_SIZE/MOD_TNUM, MOD_TNUM, 0, *aStream>>>(inBuff, outBuff, myMap);

  cufftSetStream(*myPlan, *aStream);
  cufftExecC2C(*myPlan, (cufftComplex *)outBuff, (cufftComplex *)outBuff, CUFFT_INVERSE);
}

