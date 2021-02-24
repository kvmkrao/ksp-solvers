
# ************ Author: VMK Kotteda **************
# ************ date: Feb 24, 2021  **************

# Install TPL such as mpi, boost, blas, lapack, and netcdf 

# install boost (ubuntu)
#sudo apt-get install libboost-mpi-dev 

# install netcdf (ubuntu) 
#sudo apt-get install libnetcdf-mpi-dev

# install blas lapack (ubuntu)
#sudo apt-get install libblas-dev liblapack-dev


# Modify these paths for your system.
HOME=/home/drt/
TRILINSTALLDIR=$HOME/Software/solve/Trilinos/
# Remove the CMake cache. For an extra clean start in an already-used build
# directory, rm -rf CMake* to get rid of all CMake-generated files.
rm -f CMake*;


cmake \
 -D CMAKE_INSTALL_PREFIX:PATH=${TRILINSTALLDIR}/install \
 -D CMAKE_BUILD_TYPE:STRING=RELEASE \
 -D BUILD_SHARED_LIBS:BOOL=OFF \
 -D TPL_ENABLE_MPI:BOOL=ON \
 -D Trilinos_ENABLE_Fortran:BOOL=ON \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
 -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
 -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="" \
 -D Teuchos_ENABLE_LONG_LONG_INT:BOOL=ON \
\
 -D Trilinos_ENABLE_Teuchos:BOOL=ON \
 -D Trilinos_ENABLE_Belos:BOOL=ON \
\
 -D TPL_ENABLE_Boost:BOOL=ON \
 -D TPL_ENABLE_BoostLib:BOOL=ON \
\
 -D TPL_ENABLE_Netcdf:BOOL=ON \
\
 -D Trilinos_ENABLE_Tpetra:BOOL=ON \
 -D Trilinos_ENABLE_Kokkos:BOOL=ON \
 -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
 -D Trilinos_ENABLE_Amesos2:BOOL=ON \
 -D Trilinos_ENABLE_Zoltan2:BOOL=ON \
 -D Trilinos_ENABLE_MueLu:BOOL=ON \
 -D Amesos2_ENABLE_KLU2:BOOL=ON \
  -D TPL_ENABLE_BLAS:BOOL=ON \
  -D TPL_ENABLE_LAPACK:BOOL=ON \
\
 -D Trilinos_ENABLE_Kokkos:BOOL=ON \
 -D Trilinos_ENABLE_KokkosCore:BOOL=ON \
 -D Kokkos_ENABLE_SERIAL:BOOL=OFF \
 -D Kokkos_ENABLE_OPENMP:BOOL=ON \
 -D Kokkos_ENABLE_CUDA:BOOL=OFF \
 -D Tpetra_ENABLE_CUDA:BOOL=OFF \
 -D Tpetra_INST_OPENMP:BOOL=ON \
 -D Trilinos_ENABLE_OpenMP:BOOL=ON \
 -D Kokkos_ENABLE_Pthread:BOOL=OFF \
  -D TPL_ENABLE_BLAS:BOOL=ON \
  -D TPL_ENABLE_LAPACK:BOOL=ON \
  -D CMAKE_C_FLAGS:STRING="-O3 -fPIC" \
  -D CMAKE_CXX_FLAGS:STRING=" -std=c++14 -O3 -fPIC" \
  -D CMAKE_Fortran_FLAGS:STRING=" -O3 -fPIC" \
   -D Trilinos_ENABLE_Fortran:BOOL=ON \
${TRILINSTALLDIR}/src

#Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \

make -j4
make install

