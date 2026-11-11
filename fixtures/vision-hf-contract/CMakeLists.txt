cmake_minimum_required(VERSION 3.20)
project(dinf_vision_oracle_hf_v1 LANGUAGES C CXX)
# Build only the patched mtmd library against explicit, matching installed libs.
foreach(required LLAMA_SOURCE LLAMA_PREFIX GGML_PREFIX)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Set ${required} to an absolute path")
    endif()
endforeach()
set(BUILD_SHARED_LIBS ON)
set(LLAMA_INSTALL_VERSION 0.0.8920)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_BUILD_RPATH "${LLAMA_PREFIX}/lib;${GGML_PREFIX}/lib")
find_package(ggml 0.10.0 EXACT REQUIRED CONFIG
    PATHS "${GGML_PREFIX}/lib/cmake/ggml" NO_DEFAULT_PATH)
find_library(LLAMA_LIBRARY NAMES llama PATHS "${LLAMA_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(ggml ALIAS ggml::ggml)
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION "${LLAMA_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_PREFIX}/include")
add_subdirectory("${LLAMA_SOURCE}/tools/mtmd" mtmd)
target_link_libraries(mtmd PUBLIC ggml::ggml-base)
target_include_directories(mtmd SYSTEM PRIVATE "${GGML_PREFIX}/include")
