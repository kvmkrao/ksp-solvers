
file1=read_MatrixMarket.cpp
file3=itersolve.cpp
file2=mmio.cpp
mpicc -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3    -I/home/drt/Software/solve/petsc_built_third/petsc/include -I/home/drt/Software/solve/petsc_built_third/petsc/arch-linux-c-debug/include     $file1 $file2 $file3  -Wl,-rpath,/home/drt/Software/solve/petsc_built_third/petsc/arch-linux-c-debug/lib -L/home/drt/Software/solve/petsc_built_third/petsc/arch-linux-c-debug/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -lpetsc -llapack -lblas -lX11 -lm -lstdc++ -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl -o petscsolv.exe

# run
# data url: https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/nnceng/hor__131.html
./petscsolv.exe hor__131.mtx 

