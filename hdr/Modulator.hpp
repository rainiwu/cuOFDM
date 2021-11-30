#ifndef Modulator_h
#define Modulator_h

#include "Constellation.hpp"
#include <cuComplex.h>
#include <vector>

namespace cuOFDM {

/** Modulator implements narrowband modulation
 * The Modulator class is a functor which takes some input bitstream and changes
 * it to the appropriate narrowband constellation as configured.
 */
template <typename modType> class Modulator {
public:
  Modulator();
  Modulator(const Modulator<modType> &aCopy);
  Modulator &operator=(const Modulator<modType> &aCopy);
  ~Modulator();

  void operator()(const std::vector<uint8_t> &inBits);

protected:
  // constellation map
  modType myConst;

  // complex number array with the map on cuda device
  cuComplex *dMap = nullptr;
  // modulated output on device
  cuComplex *dOut = nullptr;
  // bitstream input on device
  uint8_t *dIn = nullptr;
};

} // namespace cuOFDM

#endif
