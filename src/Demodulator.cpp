#include "Demodulator.hpp"
#include <cuda_runtime.h>
#include <fftw3.h>
#include <fstream>
#include <iostream>

namespace cuOFDM {

template <> Demodulator<QAM256>::Demodulator() : myConst() {
  cudaMalloc((void **)&dMap, sizeof(cuComplex) * myConst.getMap().size());

  cuComplex temp[myConst.getMap().size()];
  for (auto val : myConst.getMap())
    temp[val.first] = make_float2(val.second.real(), val.second.imag());

  cudaMemcpy(dMap, temp, sizeof(cuComplex) * myConst.getMap().size(),
             cudaMemcpyHostToDevice);
}

template <> Demodulator<QAM256>::~Demodulator() { cudaFree(dMap); }

template <>
void Demodulator<QAM256>::operator<<(
    const std::vector<std::complex<float>> &someMods) {
  auto output = std::array<std::complex<float>, BATCH_SIZE>();
  for (size_t i = 0; i < BATCH_SIZE; i++)
    output[i] = someMods[i];
  modQueue.push(output);
}

template <> void Demodulator<QAM256>::cpuDemod() {
  auto modBatch = modQueue.front();
  modQueue.pop();
  auto bitBatch = std::array<uint8_t, BATCH_SIZE>();

  // perform fft to get frequency domain symbols
  fftw_complex *out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * BATCH_SIZE);
  fftw_complex *in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * BATCH_SIZE);

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    in[i][0] = modBatch[i].real();
    in[i][1] = modBatch[i].imag();
  }

  auto plan_forward =
      fftw_plan_dft_1d(BATCH_SIZE, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan_forward);

  for (size_t i = 0; i < BATCH_SIZE; i++) {
    modBatch[i].real(out[i][0] / BATCH_SIZE);
    modBatch[i].imag(out[i][1] / BATCH_SIZE);
  }

  // now modbatch contains frequency domain modulated symbols
  // for each symbol, find corresponding byte
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    uint8_t j = 0;
    do {
      if (modBatch[i] == myConst.getMap().at(j) ||
          (std::abs(modBatch[i] - myConst.getMap().at(j)) < 0.01)) {
        bitBatch[i] = j;
        break;
      }
    } while (++j != 0);
  }
  bitQueue.push(bitBatch);
}

template <>
std::shared_ptr<std::array<uint8_t, BATCH_SIZE>>
Demodulator<QAM256>::getBatch() {
  auto output =
      std::make_shared<std::array<uint8_t, BATCH_SIZE>>(bitQueue.front());
  bitQueue.pop();
  return output;
}

} // namespace cuOFDM
