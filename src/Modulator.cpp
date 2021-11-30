#include "Modulator.hpp"

namespace cuOFDM {

template <> Modulator<Constellation>::Modulator() : myConst(){};

template <>
Modulator<Constellation>::Modulator(const Modulator<Constellation> &aCopy) {
  this->myConst = aCopy.myConst;
};

template <>
Modulator<Constellation> &
Modulator<Constellation>::operator=(const Modulator<Constellation> &aCopy) {
  this->myConst = aCopy.myConst;
  return *this;
}

template <> Modulator<Constellation>::~Modulator() {}

template <>
void Modulator<Constellation>::operator()(const std::vector<uint8_t> &inBits) {}

} // namespace cuOFDM
