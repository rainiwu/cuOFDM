#ifndef Modulator_h
#define Modulator_h

#include "Constellation.hpp"

namespace cuOFDM {

/** Modulator implements narrowband modulation
 * The Modulator class is a functor which takes some input bitstream and changes
 * it to the appropriate narrowband constellation as configured.
 */
class Modulator {
public:
  Modulator();
  Modulator(const Modulator &aCopy);
  Modulator &operator=(const Modulator &aCopy);
  ~Modulator();

  void operator()();

protected:
  Constellation myConst;
};

} // namespace cuOFDM

#endif
