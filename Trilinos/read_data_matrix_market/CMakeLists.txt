cmake_minimum_required(VERSION 2.8)
set(TRILINOS_PATH /home/drt/Software/solve/Trilinos/install)

FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})

IF(NOT CMAKE_CXX_COMPILER)
#  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} "-std=c++11 --expt-extended-lambda")
  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} "-std=c++11")
#  SET(CMAKE_CXX_FLAGS)
ENDIF()

IF(NOT CMAKE_C_COMPILER)
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

IF(NOT CMAKE_FORTRAN_COMPILER)
  SET(CMAKE_FORTRAN_COMPILER ${Trilinos_FORTRAN_COMPILER})
ENDIF()

#enable_language(Fortran)

project(example)

set(SOURCES 
solve_ifpack_CG.cpp)

include_directories(".")
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

# set the path to the library folder
link_directories(${PROJECT_SOURCE_DIR})

add_executable(example ${SOURCES})
target_link_libraries(example
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES})
