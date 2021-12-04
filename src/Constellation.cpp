#include "Constellation.hpp"

namespace cuOFDM {

/*******************************/

Constellation::Constellation() { genMap(); }

Constellation::Constellation(const Constellation &aCopy) { *this = aCopy; }

Constellation &Constellation::operator=(const Constellation &aCopy) {
  this->myMap = aCopy.getMap();
  return *this;
}

Constellation::~Constellation() {}

void Constellation::genMap() {}

/*******************************/

QAM256::QAM256() { genMap(); }

QAM256::QAM256(const QAM256 &aCopy) { *this = aCopy; }

QAM256 &QAM256::operator=(const QAM256 &aCopy) {
  this->myMap = aCopy.getMap();
  return *this;
}

QAM256::~QAM256() {}

void QAM256::genMap() {
  uint8_t i = 0;
  do {
    auto j = i % 64;
    auto val = std::complex<float>(j % 8 + 0.5, (uint8_t)(j / 8) + 0.5);
    val = std::complex<float>(val.real() / 7.5, val.imag() / 7.5);
    auto k = i >> 6;
    switch (k) {
    case 0:
      break;
    case 1:
      val = std::complex<float>(-val.real(), val.imag());
      break;
    case 2:
      val = std::complex<float>(-val.real(), -val.imag());
      break;
    case 3:
      val = std::complex<float>(val.real(), -val.imag());
      break;
    }
    myMap.emplace(i, val);
    i++;
  } while (i != 0);
}

/*******************************/
} // namespace cuOFDM
