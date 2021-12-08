#include "Randomizer.cuh"

__global__ void cuInitRand(curandState *aState) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(clock() + tid, tid, 0, &aState[tid]); 
}

__global__ void cuApplyRand(cuComplex* inBuff, cuComplex* outBuff, curandState *aState) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // generate rand
  float randx = curand_normal(&aState[tid]);
  float randy = curand_normal(&aState[tid]);

  randx = randx * ((float)BATCH_SIZE / 2048) * 2/3;
  randy = randy * ((float)BATCH_SIZE / 2048) * 2/3;

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
