# cuOFDM

> CUDA for digital communications using the OFDM

This project, `cuOFDM`, implements an OFDM transmit/receive pipeline in CUDA.
Currently, `cuOFDM` transmits, receives, and applies noise to multiplexed QAM256 symbols.
The current pipeline, tested on an `NVIDIA GTX 1060 6GB` machine, can achieve 100Mbps loopback throughput. 
`cuOFDM` is a work in progress - next steps include subcarrier scheduling, cyclic prefixing, and channel coding.
Direct integration with common software defined radio (SDR) heads will also be explored.

## File Structure
* `hdr/` - Header files
* `src/` - Source files
* `CMakeLists.txt` - CMake build configurations
* `README.md` - This file

## Prerequisites
1. CMake 3.12 or higher
2. NVIDIA CUDA compiler
3. C++17 or higher

## Build Instructions
The following instructions are for Linux distributions.

1. Navigate to project root
2. `mkdir build` - Create build directory 
3. `cd build` - Navigate to build directory 
4. `cmake ..` - Run CMake setup
5. `make -j` - Build project

Executable `cuOFDM` will be available in `build/src/` after `make` completes.

## Use Instructions
The produced executable runs several tests by default. 
Run the executable with `-h` or `--help` to learn about more specific run options.

