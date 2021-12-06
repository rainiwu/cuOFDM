#include "Tester.hpp"

using namespace cuOFDM;

int main(int argc, const char *argv[]) {
  if (argc > 1) {
    auto command = std::string(argv[1]);
    if ("loopback" == command || "0" == command)
      Tester::modDemod();
    else if ("--help" == command || "-h" == command)
      std::cout << "Valid arguments are:\n\t[0] loopback\n" << std::flush;
  } else {
    std::cout << "invalid argument" << std::endl;
  }
  return 0;
}
