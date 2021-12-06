#include "Tester.hpp"

using namespace cuOFDM;

int main(int argc, const char *argv[]) {
  if (argc > 1) {
    auto command = std::string(argv[1]);
    if ("loopback" == command || "0" == command)
      Tester::modDemod();
    else if ("noisy_loopback" == command || "1" == command)
      Tester::modRandDemod();
    else if ("--help" == command || "-h" == command)
      std::cout
          << "Valid arguments are:\n\t[0] loopback\n\t[1] noisy_loopback\n"
          << std::flush;
  } else {
    std::cout << "invalid argument" << std::endl;
  }
  return 0;
}
