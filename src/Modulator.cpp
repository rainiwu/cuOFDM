#include "Modulator.hpp"
#include "Modulator.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace cuOFDM {

template <> Modulator<QAM256>::Modulator() : myConst() {
  cudaMalloc((void **)&dMap, sizeof(cuComplex) * myConst.getMap().size());
  // note that QAM256 Map will be constant throughout lifetime
  // thus, we can copy it over in constructor
  cuComplex *temp =
      (cuComplex *)malloc(sizeof(cuComplex) * myConst.getMap().size());
  for (auto val : myConst.getMap()) {
    size_t i = val.first;
    temp[i] = make_float2(val.second.real(), val.second.imag());
  }
  cudaMemcpy(dMap, temp, sizeof(cuComplex) * myConst.getMap().size(),
             cudaMemcpyHostToDevice);
}

template <> Modulator<QAM256>::~Modulator() {
  // free cuda device memory
  cudaFree(dMap);
}

template <>
void Modulator<QAM256>::operator()(void *inBuff, void *outBuff,
                                   cudaStream_t *aStream) const {
  process((uint8_t *)inBuff, (cuComplex *)outBuff, dMap, aStream);
}

template <> size_t Modulator<QAM256>::getInBufSize() const {
  return BATCH_SIZE * sizeof(uint8_t);
}

template <> size_t Modulator<QAM256>::getOutBufSize() const {
  return BATCH_SIZE * sizeof(cuComplex);
}

} // namespace cuOFDM
