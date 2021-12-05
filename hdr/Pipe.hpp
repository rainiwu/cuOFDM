#ifndef Pipe_h
#define Pipe_h

#include <cuda_runtime.h>

namespace cuOFDM {
class Pipe {
public:
  Pipe() {}
  ~Pipe() {}
  Pipe(const Pipe &aCopy) = delete;
  Pipe &operator=(const Pipe &aCopy) = delete;

  virtual void operator()(void *inBuff, void *outBuff,
                          cudaStream_t *aStream) const {}

  // returns size of expected buffer in bytes
  virtual size_t getInBufSize() const { return 0; }
  virtual size_t getOutBufSize() const { return 0; }
};
} // namespace cuOFDM

#endif
