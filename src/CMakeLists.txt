set(TARGET cuOFDM)
add_executable(${TARGET} main.cpp Modulator.cu Modulator.cpp Constellation.cpp Demodulator.cpp)
target_link_libraries(${TARGET} fftw3)
