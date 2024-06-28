cmake_minimum_required(VERSION 3.12)
project(phonemizer.cpp)

# Set the library name
set(PHONEMIZER_LIB phonemizer)

# Set the output directory for the compiled binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Check for build type and set default to Release if not specified
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add subdirectories for dependencies and examples
add_subdirectory(ggml)
add_subdirectory(example)

# Create the static library
add_library(${PHONEMIZER_LIB} STATIC phonemizer.cpp phonemizer.h utils.h)

# Link the ggml library
target_link_libraries(${PHONEMIZER_LIB} PUBLIC ggml)

# Include the current directory for header files
target_include_directories(${PHONEMIZER_LIB} PUBLIC .)

# Set the C++ standard to C++11
target_compile_features(${PHONEMIZER_LIB} PUBLIC cxx_std_11)
