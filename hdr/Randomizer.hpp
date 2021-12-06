#ifndef Randomizer_h
#define Randomizer_h

#include "Common.hpp"
#include "Pipe.hpp"
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>

namespace cuOFDM {

/** Randomizer applies random noise to Modulated input
 *
 */
class Randomizer : public Pipe {
public:
  Randomizer();
  Randomizer(const Randomizer &aCopy) = delete;
  Randomizer &operator=(const Randomizer &aCopy) = delete;
  ~Randomizer();

  void operator()(void *inBuff, void *outBuff, cudaStream_t *aStream) const;
  size_t getInBufSize() const;
  size_t getOutBufSize() const;

protected:
  // randomizer seed
  curandState_t *dState;
};
} // namespace cuOFDM

#endif
