#include "Constellation.hpp"
#include <iostream>
int main() {
  auto var = cuOFDM::QAM256();
  auto myMap = var.getMap();
  std::cout << myMap[(uint8_t)30] << std::endl;
  return 0;
}
