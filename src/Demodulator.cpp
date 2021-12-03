#include "Demodulator.hpp"
#include <cuda_runtime.h>

namespace cuOFDM {

template <> Demodulator<QAM256>::Demodulator() : myConst() {
  cudaMalloc((void **)&dMap, sizeof(cuComplex) * myConst.getMap().size());
  cudaMalloc((void **)&dModBatch, sizeof(cuComplex) * BATCH_SIZE);
  cudaMalloc((void **)&dBitBatch, sizeof(uint8_t) * BATCH_SIZE);
  cudaMalloc((void **)&dInterp, sizeof(cuComplex) * BATCH_SIZE);

  cuComplex temp[myConst.getMap().size()];
  for (auto val : myConst.getMap())
    temp[val.first] = make_float2(val.second.real(), val.second.imag());

  cudaMemcpy(dMap, temp, sizeof(cuComplex) * myConst.getMap().size(),
             cudaMemcpyHostToDevice);
}

template <> Demodulator<QAM256>::~Demodulator() {
  cudaFree(dMap);
  cudaFree(dModBatch);
  cudaFree(dBitBatch);
  cudaFree(dInterp);
}

} // namespace cuOFDM
