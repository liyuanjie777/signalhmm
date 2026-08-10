cmake_minimum_required(VERSION 3.16)
project(MapSignal LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(zstd REQUIRED)
find_package(OpenMP REQUIRED)


add_executable(MapSignal
      src/fileio.cpp
      src/signal_map.cpp
      src/sparsetransition.cpp
      src/hmm.cpp
      src/variable.cpp
      src/hmm_gmm.cpp
      src/gaussianmixmodel.cpp
        src/include/sparsematrix.h
        src/sparsematrix.cpp
        src/include/pelt.hpp
        src/include/segment.h
        src/segment.cpp
        src/include/json.hpp
)

include_directories(/usr/include/eigen3
${CMAKE_SOURCE_DIR}/src/include
)
target_link_libraries(MapSignal PRIVATE zstd::libzstd_static)
target_link_libraries(MapSignal PRIVATE OpenMP::OpenMP_CXX)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()