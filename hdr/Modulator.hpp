#ifndef Modulator_h
#define Modulator_h

#include "Common.hpp"
#include "Constellation.hpp"
#include "Pipe.hpp"
#include <cuComplex.h>
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

  void operator()(void *inBuff, void *outBuff, cudaStream_t *aStream);
  size_t getInBufSize() const;
  size_t getOutBufSize() const;

  void operator<<(const std::vector<uint8_t> &someBits);
  std::shared_ptr<std::array<std::complex<float>, BATCH_SIZE>> getBatch();

  // verification function to process using CPU
  void cpuProcess();

protected:
  // constellation map
  const modType myConst;

  std::queue<std::array<uint8_t, BATCH_SIZE>> bitQueue;
  std::queue<std::array<std::complex<float>, BATCH_SIZE>> modQueue;

  // complex number array with the map on cuda device
  cuComplex *dMap = nullptr;
};

} // namespace cuOFDM

#endif
