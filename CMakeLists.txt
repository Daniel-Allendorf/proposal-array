cmake_minimum_required(VERSION 3.15)
project(sampling)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ftemplate-depth=100000")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")

add_library(libsampling INTERFACE)
target_include_directories(libsampling INTERFACE include/)
target_link_libraries(libsampling INTERFACE)

enable_testing()
add_subdirectory(source/tests)
