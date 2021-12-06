set(TARGET cuOFDM)
add_executable(${TARGET} main.cpp Modulator.cu Modulator.cpp Constellation.cpp Demodulator.cpp Demodulator.cu Streamer.cpp
  Tester.cpp)
target_link_libraries(${TARGET} "cufft")
