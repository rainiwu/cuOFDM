#include "Demodulator.hpp"
#include "Demodulator.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

namespace cuOFDM {

template <> Demodulator<QAM256>::Demodulator() : myConst() {
  cudaMalloc((void **)&dMap, sizeof(cuComplex) * myConst.getMap().size());

  cuComplex *temp =
      (cuComplex *)malloc(sizeof(cuComplex) * myConst.getMap().size());
  for (auto val : myConst.getMap())
    temp[val.first] = make_float2(val.second.real(), val.second.imag());

  cudaMemcpy(dMap, temp, sizeof(cuComplex) * myConst.getMap().size(),
             cudaMemcpyHostToDevice);
}

template <> Demodulator<QAM256>::~Demodulator() { cudaFree(dMap); }

template <>
void Demodulator<QAM256>::operator()(void *inBuff, void *outBuff,
                                     cudaStream_t *aStream) const {
  demod((cuComplex *)inBuff, (uint8_t *)outBuff, dMap, aStream);
}

template <> size_t Demodulator<QAM256>::getInBufSize() const {
  return BATCH_SIZE * sizeof(uint8_t);
}

template <> size_t Demodulator<QAM256>::getOutBufSize() const {
  return BATCH_SIZE * sizeof(cuComplex);
}

} // namespace cuOFDM
