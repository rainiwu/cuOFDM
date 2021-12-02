#include "Modulator.cuh"
#include <cstdio>

__global__ void modulate(uint8_t* dBits, cuComplex* dMods, cuComplex* dMap) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= BATCH_SIZE) printf("mistake in modulate"); 
  dMods[tid] = dMap[dBits[tid]];
}

void processBitBatch(uint8_t* hBitBatch, uint8_t* dBitBatch, cuComplex* hModBatch, cuComplex* dModBatch, cuComplex* dMap){
  cudaMemcpy(dBitBatch, hBitBatch, sizeof(uint8_t)*BATCH_SIZE, cudaMemcpyHostToDevice);
  modulate<<<BATCH_SIZE / TNUM, TNUM>>>(dBitBatch, dModBatch, dMap);
  cudaDeviceSynchronize();
  cudaMemcpy(hModBatch, dModBatch, sizeof(cuComplex)*BATCH_SIZE, cudaMemcpyDeviceToHost);
}
