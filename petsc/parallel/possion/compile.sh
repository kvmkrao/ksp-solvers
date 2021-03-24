
#file1=test.cpp
#file1=itersolve_stndlone_prea_adv.cpp

file1=itersolve_stndlone_prea_reuse.cpp
#file1=itersolve_stndlone_prea.cpp
#file1=itersolve_stndlone.cpp
#file2=read.cpp
#file2=itersolve.cpp 

# release
mpicc -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -O3 -march=native -mtune=native  -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -O3 -march=native -mtune=native    -I/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/include -I/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release/include -I/usr/local/cuda-11.2/include $file1  -Wl,-rpath,/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release/lib -L/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release/lib -Wl,-rpath,/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release/lib -L/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release/lib -Wl,-rpath,/usr/local/cuda-11.2/lib64 -L/usr/local/cuda-11.2/lib64 -Wl,-rpath,/home/vkotteda/Software/libraries/mpi/gcc7/aware-c/lib -L/home/vkotteda/Software/libraries/mpi/gcc7/aware-c/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/8 -L/usr/lib/gcc/x86_64-linux-gnu/8 -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -lpetsc -lHYPRE -lkokkoscontainers -lkokkoscore -llapack -lblas -lX11 -lm -lstdc++ -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl -o petscsol_rel.exe 

# debug 
mpicc -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3    -I/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/include -I/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-debug/include -I/usr/local/cuda-11.2/include  $file1  -Wl,-rpath,/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-debug/lib -L/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-debug/lib -Wl,-rpath,/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-debug/lib -L/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-debug/lib -Wl,-rpath,/usr/local/cuda-11.2/lib64 -L/usr/local/cuda-11.2/lib64 -Wl,-rpath,/home/vkotteda/Software/libraries/mpi/gcc7/aware-c/lib -L/home/vkotteda/Software/libraries/mpi/gcc7/aware-c/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/8 -L/usr/lib/gcc/x86_64-linux-gnu/8 -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -lpetsc -lHYPRE -lkokkoscontainers -lkokkoscore -llapack -lblas -lX11 -lm -lcufft -lcublas -lcudart -lcusparse -lcusolver -lstdc++ -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl -o  petscsol_deb.exe

mpicc -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g -O  -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g -O    -I/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/include -I/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release-cuda/include -I/usr/local/cuda-11.2/include     $file1  -Wl,-rpath,/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release-cuda/lib -L/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release-cuda/lib -Wl,-rpath,/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release-cuda/lib -L/home/vkotteda/Software/libraries/petsc_hypre_kokkos/src/linux-gnu-release-cuda/lib -Wl,-rpath,/usr/local/cuda-11.2/lib64 -L/usr/local/cuda-11.2/lib64 -Wl,-rpath,/home/vkotteda/Software/libraries/mpi/gcc7/aware-c/lib -L/home/vkotteda/Software/libraries/mpi/gcc7/aware-c/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/8 -L/usr/lib/gcc/x86_64-linux-gnu/8 -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -lpetsc -lHYPRE -llapack -lblas -lX11 -lm -lcufft -lcublas -lcudart -lcusparse -lcusolver -lstdc++ -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl -o ./petsc_cuda.exe 

#mpirun -np 4 ./petscsol_rel.exe -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre 

#for i in 1 2 4 8 16 32; do echo ; echo $i ; time  mpirun -np $i petscsol_deb.exe   -ksp_atol  1e-06   -ksp_rtol  1e-10 ; done ;
#echo 
#echo 
#echo 
#for i in 1 2 4 8 16 32; do echo ; echo $i ; time  mpirun -np $i petscsol_rel.exe   -ksp_atol  1e-06   -ksp_rtol  1e-10 ; done ;
