#include "Streamer.hpp"
#include "Common.hpp"
#include <array>

namespace cuOFDM {

Streamer::Streamer(std::vector<std::shared_ptr<Pipe>> &somePipes,
                   cudaStream_t *aStream) {
  myStream = aStream;

  auto aHead = std::make_shared<Head>();
  auto aTail = std::make_shared<Tail>();
  myPipes.push_back(aHead);

  cudaHostAlloc((void **)&inHMem, aHead->getInBufSize(), cudaHostAllocDefault);
  myDMem.emplace_back(nullptr);
  cudaMalloc((void **)&myDMem.back(), aHead->getOutBufSize());

  for (const auto &pipe : somePipes) {
    if (pipe->getInBufSize() == myPipes.back()->getOutBufSize()) {
      myPipes.push_back(pipe);
      myDMem.emplace_back(nullptr);
      cudaMalloc((void **)&myDMem.back(), pipe->getOutBufSize());
    } else {
      throw "incompatible pipes!";
    }
  }
  if (aTail->getInBufSize() == myPipes.back()->getOutBufSize())
    myPipes.push_back(aTail);
  else
    throw "incompatible end pipe!";

  cudaHostAlloc((void **)&outHMem, aHead->getInBufSize(), cudaHostAllocDefault);
}

Streamer::~Streamer() {
  cudaFreeHost(inHMem);
  cudaFreeHost(outHMem);

  for (auto val : myDMem)
    cudaFree(val);
}

void Streamer::operator()() {
  // start Head pipe
  (*myPipes[0])(inHMem, myDMem[0], myStream);
  // start central pipes
  for (size_t i = 1; i < myPipes.size() - 1; i++)
    (*myPipes[i])(myDMem[i - 1], myDMem[i], myStream);
  // start Tail pipe
  (*myPipes.back())(myDMem.back(), outHMem, myStream);
}

void Streamer::operator<<(const std::array<uint8_t, BATCH_SIZE> &input) {
  for (size_t i = 0; i < BATCH_SIZE; i++)
    inHMem[i] = input[i];
}

void Streamer::operator>>(std::array<uint8_t, BATCH_SIZE> &output) {
  for (size_t i = 0; i < BATCH_SIZE; i++)
    output[i] = outHMem[i];
}

/*******************************/

void Head::operator()(void *inBuff, void *outBuff,
                      cudaStream_t *aStream) const {
  cudaMemcpyAsync(outBuff, inBuff, this->getInBufSize(), cudaMemcpyHostToDevice,
                  *aStream);
}

inline size_t Head::getInBufSize() const {
  return BATCH_SIZE * sizeof(uint8_t);
}
inline size_t Head::getOutBufSize() const { return this->getInBufSize(); }

void Tail::operator()(void *inBuff, void *outBuff,
                      cudaStream_t *aStream) const {
  cudaMemcpyAsync(outBuff, inBuff, this->getInBufSize(), cudaMemcpyDeviceToHost,
                  *aStream);
}

inline size_t Tail::getInBufSize() const {
  return BATCH_SIZE * sizeof(uint8_t);
}
inline size_t Tail::getOutBufSize() const { return this->getInBufSize(); }

} // namespace cuOFDM
