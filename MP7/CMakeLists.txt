cmake_minimum_required(VERSION 3.8)
project(MP7)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(MP7 main.cpp)
target_link_libraries(MP7 PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(MP7 PRIVATE ${OpenCV_INCLUDE_DIRS})