
#file=amgx_mpi_poisson5pt_repeat.c
#file=amgx_mpi_poisson5pt.c
#file=amgx_mpi_capi_timers_rowcolpnt.c
file=amgx_mpi_capi_timers_repeat.c
#file=amgx_mpi_capi_timers.c
#file=madeup1/amgx_mpi_capi_multi.c
#file=madeup1/amgx_mpi_capi.c
#file=madeup1/amgx_mpi_capi_agg.c
#file=madeup1/amgx_mpi_capi_cla.c

#exe=amgx_mpi_possion5pt.exe 
exe=amgx_mpi_capi_timers_repeat.exe 
#exe=amgx_mpi_capi_timers.exe
#exe=amgx_mpi_capi_multi.exe
#exe=amgx_mpi_capi.exe
#exe=amgx_mpi_capi_agg.exe 
#exe=amgx_mpi_capi_cla.exe
PETSRC=/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/
AMGINST=/home/vkotteda/Software/libraries/Nvidia-AmgX/ 
AMGWSRC=/home/vkotteda/Software/solve/iterateS/AmgXWrapper/src/
CUDAL=/usr/
#CUDAL=/usr/local/cuda-11.2/
SRC=/home/vkotteda/Software/solve/iterateS/AmgX/compile/madeup1/src/

rm *.o 
mpicc  -I$PETSRC/linux-gnu-release/include -I$PETSRC/include -I$AMGINST/include -I$CUDAL/include -I$AMGWSRC -I$SRC  -O3 -DNDEBUG   -o amgx_capi.c.o   -c $file

mpicxx -I$PETSRC/linux-gnu-release/include -I$PETSRC/include  -I$AMGINST/include -I$CUDAL/include  -I$AMGWSRC -std=c++11 -O3 -DNDEBUG   -o init.cpp.o -c $AMGWSRC/init.cpp
#
mpicxx  -I$PETSRC/linux-gnu-release/include -I$PETSRC/include -I$AMGINST/include -I$CUDAL/include -I$AMGWSRC -I$SRC  -std=c++11 -O3 -DNDEBUG   -o misc.cpp.o -c $AMGWSRC/misc.cpp
#
mpicxx   -I$PETSRC/linux-gnu-release/include -I$PETSRC/include -I$AMGINST/include -I$CUDAL/include -I$AMGWSRC -I$SRC  -std=c++11 -O3 -DNDEBUG   -o setA.cpp.o -c $AMGWSRC/setA.cpp

mpicxx   -I$PETSRC/linux-gnu-release/include -I$PETSRC/include -I$AMGINST/include -I$CUDAL/include -I$AMGWSRC -I$SRC  -std=c++11 -O3 -DNDEBUG   -o solve.cpp.o -c $AMGWSRC/solve.cpp
#
mpicxx   -I$PETSRC/linux-gnu-release/include -I$PETSRC/include -I$AMGINST/include -I$CUDAL/include -I$AMGWSRC -I$SRC  -std=c++11 -O3 -DNDEBUG   -o AmgXSolver.cpp.o -c $AMGWSRC/AmgXSolver.cpp

/usr/bin/nvcc   -I$PETSRC/linux-gnu-release/include -I$PETSRC/include -I$AMGINST/include -I$CUDAL/include -I$AMGWSRC -I$SRC  -O3 -DNDEBUG   -x cu -c $AMGWSRC/consolidate.cu -o consolidate.cu.o

mpicxx   -std=c++11 -O3 -DNDEBUG  -rdynamic amgx_capi.c.o init.cpp.o misc.cpp.o setA.cpp.o solve.cpp.o AmgXSolver.cpp.o consolidate.cu.o  -o $exe  -L$CUDAL/lib64  -L/usr/lib/x86_64-linux-gnu/stubs  -Wl,-rpath,$CUDAL/lib64:$PETSRClinux-gnu-release/lib:$AMGINST/lib $PETSRC/linux-gnu-release/lib/libpetsc.so $AMGINST/lib/libamgxsh.so /usr/lib/x86_64-linux-gnu/libcudart_static.a -ldl -lrt -lcudadevrt -lcudart_static -lrt -ldl


