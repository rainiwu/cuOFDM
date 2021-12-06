#include "Randomizer.hpp"
#include "Randomizer.cuh"

namespace cuOFDM {

Randomizer::Randomizer() {
  cudaMalloc((void **)&dState, sizeof(curandState) * BATCH_SIZE);
  initRand(dState);
}

Randomizer::~Randomizer() { cudaFree(dState); }

void Randomizer::operator()(void *inBuff, void *outBuff,
                            cudaStream_t *aStream) const {
  applyRand((cuComplex *)inBuff, (cuComplex *)outBuff, dState, aStream);
}

size_t Randomizer::getInBufSize() const {
  return BATCH_SIZE * sizeof(cuComplex);
}

size_t Randomizer::getOutBufSize() const { return this->getInBufSize(); }

} // namespace cuOFDM
