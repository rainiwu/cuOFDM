#include "Demodulator.cuh"
#include <cufft.h>

__global__ void demap(cuComplex* dMods, uint8_t* dBits, cuComplex* dMap) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // revert scaling
  dMods[tid].x = dMods[tid].x / BATCH_SIZE;
  dMods[tid].y = dMods[tid].y / BATCH_SIZE;

  uint8_t i = 0;
  do {
    if((dMods[tid].x == dMap[i].x && dMods[tid].y == dMap[i].y) ||
        (dMods[tid].x - dMap[i].x < 0.01 && dMods[tid].y - dMap[i].y < 0.01)) {
      dBits[tid] = i;
      break;
    }
  } while (++i != 0);

}

void demod(cuComplex* inBuff, uint8_t *outBuff, cuComplex* dMap, cudaStream_t* aStream) {
  cufftHandle plan;
  cufftPlan1d(&plan, BATCH_SIZE, CUFFT_C2C, 1);
  cufftSetStream(plan, *aStream);
  cufftExecC2C(plan, (cufftComplex*)inBuff, (cufftComplex*)inBuff, CUFFT_FORWARD);
  cufftDestroy(plan);

  demap<<<BATCH_SIZE/DEMOD_TNUM, DEMOD_TNUM, 0, *aStream>>>(inBuff, outBuff, dMap);
}
