cmake_minimum_required(VERSION 3.5)
project(program)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

set(SOURCE_FILES main.cpp)
add_executable(program ${SOURCE_FILES})

target_link_libraries( program ${OpenCV_LIBS} )
