#ifndef Demodulator_h
#define Demodulator_h

#include "Common.hpp"
#include "Constellation.hpp"
#include "Pipe.hpp"
#include <cuComplex.h>
#include <memory>
#include <queue>
#include <vector>

namespace cuOFDM {

/** Demodulator implements narrowband demodulation
 * The Demodulator class is a functor which takes some input complex samples in
 * a given modulation alphabet and outputs the appropriate bitstream.
 */
template <typename modType> class Demodulator : public Pipe {
public:
  Demodulator();
  Demodulator(const Demodulator &aCopy) = delete;
  Demodulator &operator=(const Demodulator &aCopy) = delete;
  ~Demodulator();

  void operator()(void *inBuff, void *outBuff, cudaStream_t *aStream) const;
  size_t getInBufSize() const;
  size_t getOutBufSize() const;

protected:
  // constellation map
  const modType myConst;

  // complex number array with the map on cuda device
  cuComplex *dMap = nullptr;
};
} // namespace cuOFDM

#endif
