project(DiskFit)

cmake_minimum_required(VERSION 2.6)


set(CMAKE_CXX_FLAGS "-g -Wall -O3 -funroll-loops -ftree-vectorize -msse3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/undvc_common/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")



find_package(Boost REQUIRED COMPONENTS filesystem system)


add_subdirectory(tao)
add_subdirectory(undvc_common)
include_directories(${PROJECT_SOURCE_DIR}/tao)
include_directories(${PROJECT_SOURCE_DIR}/undvc_common)

#set(INCLUDES    )

set(SOURCES     Density_to_StarCounts.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})


link_directories(${Boost_LIBRARY_DIRS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} asynchronous_algorithms synchronous_algorithms tao_util undvc_common ${Boost_LIBRARIES})
