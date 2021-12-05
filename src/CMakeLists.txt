set(TARGET cuOFDM)
add_executable(${TARGET} main.cpp Modulator.cu Modulator.cpp Constellation.cpp Demodulator.cpp Streamer.cpp)
