cmake_minimum_required(VERSION 2.8)
set(TRILINOS_PATH /home/vkotteda/Software/libraries/Trilinos/mpi-gcc75/install)

FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})

IF(NOT CMAKE_CXX_COMPILER)
#  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} "-std=c++11 --expt-extended-lambda")
   SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} "-std=c++11")
ENDIF()

IF(NOT CMAKE_C_COMPILER)
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

IF(NOT CMAKE_FORTRAN_COMPILER)
  SET(CMAKE_FORTRAN_COMPILER ${Trilinos_FORTRAN_COMPILER})
ENDIF()

SET(CUDA_DIR /usr/local/cuda-11.2/)
SET(LAPACK_DIR /home/vkotteda/Software/libraries/lapack-v3.9.0/gcc7/install/)

## =====================================================================
#
#MESSAGE("")
#MESSAGE("-- Finding CUDA")
#
#SET(CUDA_TOOLKIT_ROOT_DIR ${CUDA_DIR} CACHE PATH "" FORCE)
#
##FIND_PACKAGE(CUDA REQUIRED)
#
#IF(${CUDA_FOUND})
#    SET(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
#    MESSAGE("-- Finding CUDA - Success")
#ENDIF()
## =====================================================================


#enable_language(Fortran)

project(example)

set(SOURCES 
tpetra_belos_ifpack2.cpp)
#tpetra_belos_muelu.cpp) 

include_directories(".")
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${CUDA_DIR}/include)
link_directories(/home/vkotteda/Software/libraries/lapack-v3.9.0/gcc7/install/include)

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})
LINK_DIRECTORIES(${CUDA_DIR}/lib64) 
set(PROJECT_LINK_LIBS libblas.so liblapack.so)
link_directories(/home/vkotteda/Software/libraries/lapack-v3.9.0/gcc7/install/lib/)


# set the path to the library folder
link_directories(${PROJECT_SOURCE_DIR})

add_executable(example ${SOURCES})
target_link_libraries(example
  ${Trilinos_LIBRARIES} 
  ${Trilinos_TPL_LIBRARIES}
  -lteuchoskokkoscomm
  -lcudart
  -lblas -llapack
  )

