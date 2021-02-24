
cmake_minimum_required(VERSION 2.8)
#set(EXE serial_CG )
#set (Trilinos_DIR /pylon2/ac4s8rp/kotteda/cuda-gcc/install)
set(TRILINOS_PATH /ascldap/users/vkkotte/Trilinos/mpi/installnew)
set(CMAKE_PREFIX_PATH ${Trilinos_PATH} ${CMAKE_PREFIX_PATH})
#find_package(Trilinos REQUIRED)

FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})
#FIND_PACKAGE(Trilinos PATHS /pylon2/ac4s8rp/kotteda/cuda-gcc/install/lib/cmake/Trilinos ${TRILINOS_PATH})
#set(CMAKE_PREFIX_PATH ${Trilinos_PATH} ${CMAKE_PREFIX_PATH})

#find_package(Trilinos REQUIRED)

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

project(serial_MueLu)

set(SOURCES
solve_muelu.cpp) 
#05_solve.cpp )
#wrapTpetraSolver.cpp )
#Ifpack2_UnitTestLocalSparseTriangularSolver.cpp

#05_solve.cpp )
#02_problem.cpp ) 
#CWrapper.cpp  FWrapper.f90 readxml.cpp )


include_directories(".")
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

# set the path to the library folder
link_directories(${PROJECT_SOURCE_DIR})


add_executable(serial_MueLu  ${SOURCES})
target_link_libraries(serial_MueLu
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES})

