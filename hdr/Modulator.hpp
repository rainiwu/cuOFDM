#ifndef Modulator_h
#define Modulator_h

#include "Common.hpp"
#include "Constellation.hpp"
#include "Pipe.hpp"
#include <cuComplex.h>
#include <cufft.h>
#include <memory>
#include <queue>
#include <vector>

namespace cuOFDM {

/** Modulator implements narrowband modulation
 * The Modulator class is a functor which takes some input bitstream and changes
 * it to the appropriate narrowband constellation as configured.
 */
template <typename modType> class Modulator : public Pipe {
public:
  Modulator();
  Modulator(const Modulator<modType> &aCopy) = delete;
  Modulator &operator=(const Modulator<modType> &aCopy) = delete;
  ~Modulator();

  void operator()(void *inBuff, void *outBuff, cudaStream_t *aStream) const;
  size_t getInBufSize() const;
  size_t getOutBufSize() const;

protected:
  // constellation map
  const modType myConst;

  // complex number array with the map on cuda device
  cuComplex *dMap = nullptr;
  // fft planner
  cufftHandle *dPlan = nullptr;
};

} // namespace cuOFDM

#endif
