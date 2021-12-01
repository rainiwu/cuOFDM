set(TARGET cuOFDM)
add_executable(${TARGET} main.cpp test.cu Modulator.cpp Constellation.cpp)
