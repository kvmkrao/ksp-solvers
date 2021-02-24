# install boost (ubuntu)
#sudo apt-get install libboost-mpi-dev 

# install netcdf (ubuntu) 
#sudo apt-get install libnetcdf-mpi-dev

# install blas lapack (ubuntu)
#sudo apt-get install libblas-dev liblapack-dev

#install metis and parmetis 
#sudo apt-get install  libmetis-dev libparmetis-dev

/configure --CFLAGS='-O3' --CXXFLAGS='-O3' --FFLAGS='-O3' --with-debugging=no --download-mpich=yes --download-hdf5=yes --download-fblaslapack=yes --download-metis=yes --download-parmetis=yes

#./configure 
#make PETSC_DIR=$HOME/Software/solve/petsc_built_third/petsc PETSC_ARCH=arch-linux-c-debug all
