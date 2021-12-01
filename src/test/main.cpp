#include "Constellation.hpp"
#include <iostream>
int main() {
  auto var = cuOFDM::QAM256();
  auto myMap = var.getMap();
  uint8_t index;
  std::string line;
  while (std::getline(std::cin, line)) {
    index = (uint8_t)std::stoi(line);
    std::cout << myMap[index] << std::endl;
  }
  return 0;
}
