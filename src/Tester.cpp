#include "Tester.hpp"

namespace cuOFDM {
namespace Tester {

void makeRand(std::array<uint8_t, BATCH_SIZE> &output) {
  std::random_device aDev;
  std::mt19937 me{aDev()};
  std::uniform_int_distribution<int> dist{0, 255};
  auto gen = [&dist, &me]() { return dist(me); };
  std::generate(output.begin(), output.end(), gen);
}

int modDemod() {
  auto in = std::array<uint8_t, BATCH_SIZE>();

  makeRand(in);

  auto pipeline = std::vector<std::shared_ptr<Pipe>>();
  auto mod = std::make_shared<Modulator<QAM256>>();
  auto demod = std::make_shared<Demodulator<QAM256>>();

  cudaStream_t aStream;
  cudaStreamCreate(&aStream);

  pipeline.push_back(mod);
  pipeline.push_back(demod);

  auto top = std::make_shared<Streamer>(pipeline, &aStream);
  (*top) << in;

  (*top)();
  cudaStreamSynchronize(aStream);

  auto out = std::array<uint8_t, BATCH_SIZE>();
  (*top) >> out;

  auto result = std::string("PASS");
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    if (in[i] != out[i]) {
      result = std::string("FAIL");
      break;
    }
  }

  std::cout << "loopback test: " << result << std::endl;

  if (std::string("PASS") == result)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

int pipedModDemod() {
  auto in = std::vector<std::array<uint8_t, BATCH_SIZE>>(PIPES);

  auto pipeline = std::vector<std::shared_ptr<Pipe>>();
  auto mod = std::make_shared<Modulator<QAM256>>();
  auto demod = std::make_shared<Demodulator<QAM256>>();

  pipeline.push_back(mod);
  pipeline.push_back(demod);

  auto result = std::string("PASS");
  auto out = std::array<uint8_t, BATCH_SIZE>();
  auto top = std::vector<std::shared_ptr<Streamer>>(PIPES);

  // prep pipes
  cudaStream_t aStream[PIPES];
  for (size_t i = 0; i < PIPES; i++) {
    cudaStreamCreate(&aStream[i]);
    makeRand(in[i]);
  }

  // run pipes
  for (size_t i = 0; i < PIPES; i++) {
    top[i] = std::make_shared<Streamer>(pipeline, &aStream[i]);
    (*top[i]) << in[i];
    (*top[i])();
  }

  // flush pipes
  for (size_t i = 0; i < PIPES; i++) {
    cudaStreamSynchronize(aStream[i]);
    (*top[i]) >> out;
    for (size_t j = 0; j < BATCH_SIZE; j++) {
      if (in[i][j] != out[j]) {
        result = std::string("FAIL");
        break;
      }
    }
  }

  std::cout << "piped loopback test: " << result << std::endl;

  if (std::string("PASS") == result)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

int modRandDemod() {
  auto in = std::array<uint8_t, BATCH_SIZE>();

  makeRand(in);

  auto pipeline = std::vector<std::shared_ptr<Pipe>>();
  auto mod = std::make_shared<Modulator<QAM256>>();
  auto rand = std::make_shared<Randomizer>();
  auto demod = std::make_shared<Demodulator<QAM256>>();

  cudaStream_t aStream;
  cudaStreamCreate(&aStream);

  pipeline.push_back(mod);
  pipeline.push_back(rand);
  pipeline.push_back(demod);

  auto top = std::make_shared<Streamer>(pipeline, &aStream);
  (*top) << in;

  (*top)();
  cudaStreamSynchronize(aStream);

  auto out = std::array<uint8_t, BATCH_SIZE>();
  (*top) >> out;

  unsigned int mismatch = 0;
  for (size_t i = 0; i < BATCH_SIZE; i++)
    if (in[i] != out[i])
      mismatch++;

  auto result = std::string("PASS");
  if (((float)mismatch) / ((float)BATCH_SIZE) > 0.5)
    result = std::string("FAIL");

  std::cout << "noisy loopback test: " << result << " with "
            << (1.0 - ((float)mismatch) / ((float)BATCH_SIZE)) * 100.0
            << "\% accuracy (" << mismatch << " mismatched bytes)" << std::endl;

  if (std::string("PASS") == result)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

int throughput() {
  auto in = std::vector<std::array<uint8_t, BATCH_SIZE>>(PIPES);

  auto pipeline = std::vector<std::shared_ptr<Pipe>>();
  auto mod = std::make_shared<Modulator<QAM256>>();
  auto rand = std::make_shared<Randomizer>();
  auto demod = std::make_shared<Demodulator<QAM256>>();

  pipeline.push_back(mod);
  pipeline.push_back(rand);
  pipeline.push_back(demod);

  auto result = std::string("PASS");
  unsigned int mismatch = 0;

  auto out = std::array<uint8_t, BATCH_SIZE>();
  auto top = std::vector<std::shared_ptr<Streamer>>(PIPES);

  auto start = std::chrono::high_resolution_clock::now();

  // prep pipes
  cudaStream_t aStream[PIPES];
  for (size_t i = 0; i < PIPES; i++) {
    cudaStreamCreate(&aStream[i]);
    makeRand(in[i]);
    top[i] = std::make_shared<Streamer>(pipeline, &aStream[i]);
  }

  unsigned int repeats = 0;
  while (repeats < REPEATS) {
    // first run
    if (0 == repeats) {
      for (size_t i = 0; i < PIPES; i++) {
        (*top[i]) << in[i];
        (*top[i])();
      }
    }

    // flush pipes and rereun
    for (size_t i = 0; i < PIPES; i++) {
      cudaStreamSynchronize(aStream[i]);
      (*top[i]) >> out;
      for (size_t j = 0; j < BATCH_SIZE; j++) {
        if (in[i][j] != out[j]) {
          mismatch++;
        }
      }
      if (repeats != REPEATS) {
        // although using the same stream, redo the copy back to simulate unique
        // dataflow - not done to avoid overusing host memory
        (*top[i]) << in[i];
        (*top[i])();
      }
    }
    repeats++;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  if (((float)mismatch) / ((float)((BATCH_SIZE) * (REPEATS * PIPES))) > 0.5)
    result = std::string("FAIL");

  // calculate throughput in Mbits/s
  // symbol count * 8 / time in us
  float throughput = ((float)(BATCH_SIZE * REPEATS * PIPES) * 8) / time.count();

  std::cout << "throughput test: " << result << " with " << throughput
            << "Mbps throughput (" << (BATCH_SIZE * REPEATS * PIPES)
            << " symbols in " << (time.count() / 1e6) << " seconds) and "
            << (1.0 -
                ((float)mismatch) / ((float)(BATCH_SIZE * REPEATS * PIPES))) *
                   100.0
            << "\% accuracy (" << mismatch << " mismatched bytes)" << std::endl;

  if (std::string("PASS") == result)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

} // namespace Tester
} // namespace cuOFDM
