#include "Modulator.hpp"
#include "test.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace cuOFDM {

template <> Modulator<QAM256>::Modulator() : myConst() {
  // initialize QAM256
  // initialize CUDA device variables
  size_t mapSize = sizeof(cuComplex) * myConst.getMap().size();
  cudaMalloc((cuComplex **)&dMap, mapSize);
  cudaMalloc((cuComplex **)&dOut, sizeof(cuComplex) * this->MAXSIZE);
  cudaMalloc((uint8_t **)&dIn, sizeof(uint8_t) * this->MAXSIZE);
  // note that QAM256 Map will be constant throughout lifetime
  // thus, we can copy it over in constructor
  cuComplex temp[myConst.getMap().size()];
  for (auto val : myConst.getMap())
    temp[val.first] = make_float2(val.second.real(), val.second.imag());
  // TODO: remove
  std::cout << "map example of 30 " << temp[30].x << ',' << temp[30].y
            << std::endl;
  cudaMemcpy(dMap, temp, mapSize, cudaMemcpyHostToDevice);
}

template <> Modulator<QAM256>::Modulator(const Modulator<QAM256> &aCopy) {
  this->myConst = aCopy.myConst;
  // TODO: implement deep copy
}

template <>
Modulator<QAM256> &
Modulator<QAM256>::operator=(const Modulator<QAM256> &aCopy) {
  this->myConst = aCopy.myConst;
  return *this;
  // TODO: implement deep copy
}

template <> Modulator<QAM256>::~Modulator() {
  // free cuda device memory
  cudaFree(dMap);
  cudaFree(dOut);
  cudaFree(dIn);
}
template <>
void Modulator<QAM256>::operator()(const std::vector<uint8_t> &inBits) {
  // copy given bitstream to device
  mySize = inBits.size();
  uint8_t temp[mySize];
  size_t i = 0;
  for (auto val : inBits)
    temp[i++] = val;
  cudaMemcpy(dIn, temp, mySize * sizeof(uint8_t), cudaMemcpyHostToDevice);
  // modulate on device
  callBitsToIq(dMap, dIn, dOut, mySize);
  // TODO: change to non-blocking
  cudaDeviceSynchronize();
  resultReady = true;
}

template <> bool Modulator<QAM256>::isReady() { return resultReady; }

template <>
std::shared_ptr<std::vector<std::complex<float>>>
Modulator<QAM256>::getResult() {
  if (false == this->isReady())
    throw "No result ready";
  cuComplex *temp = new cuComplex[mySize];
  cudaMemcpy(temp, dOut, mySize * sizeof(cuComplex), cudaMemcpyDeviceToHost);
  auto out = std::make_shared<std::vector<std::complex<float>>>(mySize);
  for (size_t i = 0; i < mySize; i++) {
    (*out)[i] = std::complex<float>(temp[i].x, temp[i].y);
  }
  delete[] temp;
  resultReady = false;
  return out;
}

} // namespace cuOFDM
