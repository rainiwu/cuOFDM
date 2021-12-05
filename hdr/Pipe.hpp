#ifndef Pipe_h
#define Pipe_h

#include <cuda_runtime.h>

namespace cuOFDM {
class Pipe {
public:
  Pipe(const Pipe &aCopy) = delete;
  Pipe &operator=(const Pipe &aCopy) = delete;

  virtual void operator()(void *inBuff, void *outBuff,
                          cudaStream_t *aStream) = 0;

  // returns size of expected buffer in bytes
  virtual size_t getInBufSize() const = 0;
  virtual size_t getOutBufSize() const = 0;

protected:
  Pipe() {}
  ~Pipe() {}
};
} // namespace cuOFDM

#endif
