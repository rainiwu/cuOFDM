#include "Modulator.hpp"
#include "Modulator.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace cuOFDM {

template <> Modulator<QAM256>::Modulator() : myConst() {
  cudaMalloc((void **)&dMap, sizeof(cuComplex) * myConst.getMap().size());
  cudaMalloc((void **)&dModBatch, sizeof(cuComplex) * BATCH_SIZE);
  cudaMalloc((void **)&dBitBatch, sizeof(uint8_t) * BATCH_SIZE);
  // note that QAM256 Map will be constant throughout lifetime
  // thus, we can copy it over in constructor
  cuComplex temp[myConst.getMap().size()];
  for (auto val : myConst.getMap())
    temp[val.first] = make_float2(val.second.real(), val.second.imag());
  cudaMemcpy(dMap, temp, sizeof(cuComplex) * myConst.getMap().size(),
             cudaMemcpyHostToDevice);
}

template <> Modulator<QAM256>::~Modulator() {
  // free cuda device memory
  cudaFree(dMap);
  cudaFree(dModBatch);
  cudaFree(dBitBatch);
}

template <>
void Modulator<QAM256>::operator<<(const std::vector<uint8_t> &someBits) {
  auto batch = std::array<uint8_t, BATCH_SIZE>();
  for (size_t i = 0; i < BATCH_SIZE; i++)
    batch[i] = someBits[i];
  bitQueue.push(batch);
  // TODO: handle nonideal case
}

template <> void Modulator<QAM256>::process() {
  // load bits into batch
  auto bitBatch = bitQueue.front();
  bitQueue.pop();
  // initialize new mod
  auto modBatch = std::array<std::complex<float>, BATCH_SIZE>();
  // process batch - assumes blocking
  processBitBatch(&bitBatch.front(), dBitBatch, (cuComplex *)&modBatch.front(),
                  dModBatch, dMap);
  modQueue.push(modBatch);
}

template <>
std::shared_ptr<std::array<std::complex<float>, BATCH_SIZE>>
Modulator<QAM256>::getBatch() {
  auto output = std::make_shared<std::array<std::complex<float>, BATCH_SIZE>>(
      modQueue.front());
  modQueue.pop();
  return output;
}

} // namespace cuOFDM
