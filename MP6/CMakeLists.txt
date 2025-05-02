cmake_minimum_required(VERSION 3.8)
project(MP6)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(MP6 main.cpp)
target_link_libraries(MP6 ${OpenCV_LIBS})
target_include_directories(MP6 PRIVATE ${OpenCV_INCLUDE_DIRS})