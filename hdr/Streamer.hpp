#ifndef Streamer_h
#define Streamer_h

#include "Pipe.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace cuOFDM {

class Streamer {
public:
  Streamer() = delete;
  Streamer(const Streamer &aCopy) = delete;
  Streamer &operator=(const Streamer &aCopy) = delete;

  Streamer(std::vector<std::shared_ptr<Pipe>> &somePipes,
           cudaStream_t *aStream);
  ~Streamer();

  void operator()();

protected:
  cudaStream_t *myStream;

  std::vector<std::shared_ptr<Pipe>> myPipes;
  std::vector<void *> myDMem;
  uint8_t *inHMem;
  uint8_t *outHMem;
};

/** Head Pipes take host pointers to uint8_t and reads into device memory
 *
 */
class Head : public Pipe {
public:
  Head() {}
  ~Head() {}

  void operator()(void *inBuff, void *outBuff, cudaStream_t *aStream) const;

  size_t getInBufSize() const;
  size_t getOutBufSize() const;
};

/** Tail pipes take a device pointer and outputs to a host pointer
 *
 */

class Tail : public Pipe {
public:
  Tail() {}
  ~Tail() {}

  void operator()(void *inBuff, void *outBuff, cudaStream_t *aStream) const;

  size_t getInBufSize() const;
  size_t getOutBufSize() const;
};

} // namespace cuOFDM

#endif
