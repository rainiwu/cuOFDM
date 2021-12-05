#include "Modulator.cuh"
#include <cufft.h>

__global__ void modulate(uint8_t* dBits, cuComplex* dMods, cuComplex* dMap) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dMods[tid] = dMap[dBits[tid]];

}

void process(uint8_t* inBuff, cuComplex* outBuff, cuComplex* myMap, cudaStream_t* aStream) {
  modulate<<<BATCH_SIZE/MOD_TNUM, MOD_TNUM, 0, *aStream>>>(inBuff, outBuff, myMap);

  cufftHandle plan;
  cufftPlan1d(&plan, BATCH_SIZE, CUFFT_C2C, 1);
  cufftSetStream(plan, *aStream);
  cufftExecC2C(plan, (cufftComplex *)outBuff, (cufftComplex *)outBuff, CUFFT_INVERSE);
  cufftDestroy(plan);
}

