#ifndef Demodulator_h
#define Demodulator_h

#include "Constellation.hpp"
#include <cuComplex.h>
#include <vector>

namespace cuOFDM {

/** Demodulator implements narrowband demodulation
 * The Demodulator class is a functor which takes some input complex samples in
 * a given modulation alphabet and outputs the appropriate bitstream.
 */
template <typename modType> class Demodulator {
public:
  Demodulator();
  Demodulator(const Demodulator &aCopy);
  Demodulator &operator=(const Demodulator &aCopy);
  ~Demodulator();

  void operator()(const std::vector<std::complex<float>>);

protected:
  // constellation map
  modType myConst;

  // complex number array with the map on cuda device
  cuComplex *dMap = nullptr;
  // bitstream output on device
  uint8_t *dOut = nullptr;
  // modulated input on device
  cuComplex *dIn = nullptr;
};
} // namespace cuOFDM

#endif
