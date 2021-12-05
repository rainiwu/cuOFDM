#include "Modulator.hpp"
#include "Modulator.cuh"
#include <cuda_runtime.h>
#include <fftw3.h>
#include <iostream>

namespace cuOFDM {

template <> Modulator<QAM256>::Modulator() : myConst() {
  cudaMalloc((void **)&dMap, sizeof(cuComplex) * myConst.getMap().size());
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
}

template <>
void Modulator<QAM256>::operator<<(const std::vector<uint8_t> &someBits) {
  auto batch = std::array<uint8_t, BATCH_SIZE>();
  for (size_t i = 0; i < BATCH_SIZE; i++)
    batch[i] = someBits[i];
  bitQueue.push(batch);
  // TODO: handle nonideal case
}

template <> void Modulator<QAM256>::cpuProcess() {
  auto bitBatch = bitQueue.front();
  bitQueue.pop();
  auto modBatch = std::array<std::complex<float>, BATCH_SIZE>();

  for (size_t i = 0; i < BATCH_SIZE; i++)
    modBatch[i] = myConst.getMap().at(bitBatch[i]);

  fftw_complex *out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * BATCH_SIZE);
  fftw_complex *in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * BATCH_SIZE);

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    in[i][0] = modBatch[i].real();
    in[i][1] = modBatch[i].imag();
  }

  auto plan_backward =
      fftw_plan_dft_1d(BATCH_SIZE, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(plan_backward);

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    modBatch[i].real(out[i][0]);
    modBatch[i].imag(out[i][1]);
  }

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
