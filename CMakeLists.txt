cmake_minimum_required(VERSION 3.16)
project(MapSignal LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(zstd REQUIRED)


add_executable(MapSignal
  src/fileio.cpp
  src/gaussianmodel.cpp
  src/histogrammodel.cpp
  src/signal_map.cpp
  src/sparsetransition.cpp
  src/hmm.cpp
  src/variable.cpp
  src/hmm_gmm.cpp
  src/gaussianmixmodel.cpp
        src/include/sparsematrix.h
        src/sparsematrix.cpp
)

include_directories(/usr/include/eigen3
${CMAKE_SOURCE_DIR}/src/include
)
target_link_libraries(MapSignal PRIVATE zstd::libzstd_static)

set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")