cmake_minimum_required(VERSION 3.12)
project(phonemizer.cpp)
set(PHONEMIZER_LIB phonemizer)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ggml)
add_library(${PHONEMIZER_LIB} STATIC phonemizer.cpp phonemizer.h utils.h)

add_subdirectory(example)

target_link_libraries(${PHONEMIZER_LIB} PUBLIC ggml)
target_include_directories(${PHONEMIZER_LIB} PUBLIC .)
target_compile_features(${PHONEMIZER_LIB} PUBLIC cxx_std_11)