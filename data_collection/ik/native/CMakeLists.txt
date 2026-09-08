cmake_minimum_required(VERSION 3.16)
project(kuavo_plantik_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KUAVO_IK_RUNTIME "$ENV{KUAVO_IK_RUNTIME}" CACHE PATH
    "External directory containing include/plantIK.h and lib/libplantIK.so")
if(NOT KUAVO_IK_RUNTIME)
  message(FATAL_ERROR "Set KUAVO_IK_RUNTIME to the external plantIK runtime root")
endif()

find_package(drake CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

set(PLANTIK_HEADER "${KUAVO_IK_RUNTIME}/include/plantIK.h")
set(PLANTIK_LIBRARY "${KUAVO_IK_RUNTIME}/lib/libplantIK.so")
if(NOT EXISTS "${PLANTIK_HEADER}" OR NOT EXISTS "${PLANTIK_LIBRARY}")
  message(FATAL_ERROR "Missing ${PLANTIK_HEADER} or ${PLANTIK_LIBRARY}")
endif()

add_executable(kuavo_plantik_server src/kuavo_plantik_server.cpp)
target_include_directories(kuavo_plantik_server PRIVATE
  "${KUAVO_IK_RUNTIME}/include"
  "${EIGEN3_INCLUDE_DIRS}"
)
target_link_libraries(kuavo_plantik_server PRIVATE
  "${PLANTIK_LIBRARY}"
  drake::drake
)
target_link_options(kuavo_plantik_server PRIVATE
  "-Wl,-rpath,${KUAVO_IK_RUNTIME}/lib"
)

