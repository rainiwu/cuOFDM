#include "Randomizer.cuh"

__global__ void cuInitRand(curandState *aState) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(clock() + tid, tid, 0, &aState[tid]); 
}

__global__ void cuApplyRand(cuComplex* inBuff, cuComplex* outBuff, curandState *aState) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // generate rand and normalize to [-0.5, 0.5]
  float randx = curand_uniform(&aState[tid]);
  randx = randx - 0.5;
  float randy = curand_uniform(&aState[tid]);
  randy = randy - 0.5;

  // apply -10dB scaling factor
  randx = randx / 2;
  randy = randy / 2;

  // apply to signal
  outBuff[tid].x = inBuff[tid].x + randx;
  outBuff[tid].y = inBuff[tid].y + randy;
}

void initRand(curandState *aState) {
  cuInitRand<<<BATCH_SIZE/RAND_TNUM, RAND_TNUM>>>(aState);
}

void applyRand(cuComplex* inBuff, cuComplex* outBuff, curandState *aState, cudaStream_t *aStream) {
  cuApplyRand<<<BATCH_SIZE/RAND_TNUM, RAND_TNUM, 0, *aStream>>>(inBuff, outBuff, aState);
}
