# =====================================================================
# \file CMakeLists.txt
# \author V Kotteda 
# \date 2021-02-04
# =====================================================================

# my cmake file for teton-project
cmake_minimum_required(VERSION 3.0) 
#SET(CMAKE_SYSTEM_NAME Generic)
# =====================================================================
# Cache variables
# =====================================================================

# set paths
SET(PETSC_DIR /home/vkotteda/Software/libraries/petsc_hypre_kokkos/src)
SET(PETSC_ARCH linux-gnu-release-cuda)
SET(BOOST_DIR /home/vkotteda/Software/libraries/boost/gcc7/install/)
SET(METIS_DIR /home/vkotteda/Software/libraries/parmetis/install/) 
SET(MPI_DIR /home/vkotteda/Software/libraries/mpi/gcc7/aware-c/) 
#SET(AMGX_DIR /home/exx/Downloads/AMGX-main)


# set default c++ compilers
SET(CMAKE_CXX_COMPILER ${MPI_DIR}/bin/mpicxx)
SET(CMAKE_C_COMPILER ${MPI_DIR}/bin/mpicc)

# set default flags for release and debug mode
# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_CXX_FLAGS -DUSE_DOUBLES=1)

# =====================================================================
# Project settings
# =====================================================================

MESSAGE("")
MESSAGE("====================================")
MESSAGE("Begin configuration")
MESSAGE("====================================")

# start generator
project(teton)

# other cmake setting
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_COLOR_MAKEFILE ON)

#find_package(MPI REQUIRED)
# =====================================================================
# AmgX
# =====================================================================

#MESSAGE("")
#MESSAGE("-- Finding AmgX")
#
#MESSAGE("-- Finding amgx_c.h")
#FIND_PATH(AMGX_INCLUDE_DIRS 
#          NAMES amgx_c.h  
#          PATHS ${AMGX_DIR} $ENV{AMGX_DIR} 
#          PATH_SUFFIXES "include" "base/include" "examples")
#
#IF(${AMGX_INCLUDE_DIRS} STREQUAL "AMGX_INCLUDE_DIRS-NOTFOUND")
#    MESSAGE(FATAL_ERROR "Could not find amgx_c.h!!!")
#ELSE()
#    MESSAGE("-- Finding amgx_c.h - found in " ${AMGX_INCLUDE_DIRS})
#ENDIF()
#
#
#MESSAGE("-- Finding amgx_config.h")
#FIND_FILE(AMGX_CONFIG_FOUND 
#          NAMES amgx_config.h 
#          PATHS ${AMGX_INCLUDE_DIRS} 
#          NO_DEFAULT_PATH)
#
#IF(${AMGX_CONFIG_FOUND} STREQUAL "AMGX_CONFIG_FOUND-NOTFOUND")
#    MESSAGE(FATAL_ERROR "amgx_config.h was not found in: " ${AMGX_INCLUDE_DIRS})
#ELSE()
#    MESSAGE("-- Finding amgx_config.h - found")
#ENDIF()
#
#
#MESSAGE("-- Finding libamgxsh.so")
#FIND_LIBRARY(AMGX_LIBRARIES
#             NAMES amgxsh
#             PATHS ${AMGX_DIR} $ENV{AMGX_DIR}
#             PATH_SUFFIXES "lib" "lib64")
#
#IF(${AMGX_LIBRARIES} STREQUAL "AMGX_LIBRARIES-NOTFOUND")
#    MESSAGE(FATAL_ERROR "libamgxsh.so was not found")
#ELSE()
#    MESSAGE("-- Finding libamgxsh.so - found: " ${AMGX_LIBRARIES})
#ENDIF()
#
#MESSAGE("-- Finding AmgX - Success")

# =====================================================================
#SET src directory path 
SET(SRC ${CMAKE_SOURCE_DIR}/src)

#set list of files to compile 
set(SOURCE_EXE
	${SRC}/main.cxx
    	${SRC}/application.cpp
    	${SRC}/operations/graph_operation.cpp
    	${SRC}/operations/linear_solver_petsc.cpp
    	${SRC}/operations/mpi_utilities.cpp
    	${SRC}/operations/uf_clustering.cpp
    	${SRC}/operations/parallel_center.cpp
    	${SRC}/operations/pressure_solver.cpp
    	${SRC}/operations/pnm_operation.cpp
    	${SRC}/pore_network/cap_displace.cpp
    	${SRC}/pore_network/cap_press_finder.cpp
    	${SRC}/pore_network/comm_info.cpp
    	${SRC}/pore_network/inlet_reservoir.cpp
    	${SRC}/pore_network/outlet_reservoir.cpp
    	${SRC}/pore_network/reservoir.cpp
    	${SRC}/pore_network/pore_network.cxx
    	${SRC}/pore_network/pore_element.cxx
    	${SRC}/pore_network/pn_iterator.cxx
    	${SRC}/pore_network/node.cxx
    	${SRC}/pore_network/network_loader.cxx
    	${SRC}/pore_network/link.cxx
    	${SRC}/pore_network/cluster.cpp
    	${SRC}/pore_network/corner_region.cpp
    	${SRC}/pore_network/curve.cpp
    	${SRC}/misc/user_input.cxx
    	${SRC}/misc/tools.cpp
    	${SRC}/misc/easylogging++.cc
    	${SRC}/misc/random_generator.cpp
    	${SRC}/misc/output.cpp
    	${SRC}/misc/my_math.cpp
    	${SRC}/misc/pchip.cpp
    	${SRC}/simulation/simulation.cpp
    	${SRC}/simulation/qss_simulation.cpp
    	${SRC}/simulation/weak_dyn_simulation.cpp)

#include header files 
include_directories(".")
#INCLUDE(GNUInstallDirs)
include_directories(${PETSC_DIR}/include
	${PETSC_DIR}/${PETSC_ARCH}/include
	${BOOST_DIR}/include
        /usr/local/include
        src/
        src/operations/
        src/pore_network/
        src/misc/
        src/simulation/)

#For the static library:
set(PROJECT_LINK_LIBS libmetis.a)
link_directories(${METIS_DIR}/lib/)

set(PROJECT_LINK_LIBS libboost_system.a  libboost_filesystem.a)
link_directories(${BOOST_DIR}/lib/)

set(PROJECT_LINK_LIBS libpetsc.so) 
LINK_DIRECTORIES(${PETSC_DIR}/${PETSC_ARCH}/lib/)

set(PROJECT_LINK_LIBS libmpi.so)
LINK_DIRECTORIES(${MPI_DIR}/lib/) 

ADD_EXECUTABLE(teton ${SOURCE_EXE})

TARGET_LINK_LIBRARIES(teton
	-lpetsc 
	-lmetis 
	-lmpi
	-lboost_filesystem -lboost_system) 

# =====================================================================
# Print information
# =====================================================================

MESSAGE("")
MESSAGE("====================================")
MESSAGE("Config Information:")
MESSAGE("====================================")
MESSAGE("")

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE("Build shared library: " ${BUILD_SHARED_LIBS})
MESSAGE("Doxygen available: " ${DOXYGEN_FOUND})
MESSAGE("Installation path: " ${CMAKE_INSTALL_PREFIX})
MESSAGE("")

MESSAGE("Compilers:")
MESSAGE("\tCXX compiler: " ${CMAKE_CXX_COMPILER})
MESSAGE("\tC compiler: " ${CMAKE_C_COMPILER})
MESSAGE("")

MESSAGE("PETSc:")
MESSAGE("\tPETSc dir: " ${PETSC_DIR})
MESSAGE("")
