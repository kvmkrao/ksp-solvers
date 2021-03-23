
for i in 1 2 4 8 16 32; do echo $i ; mpirun -np $i  ./petscsol_rel.exe -nx 100 -ny 100 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre ; mv time_out.dat  100by100_time_out_$i.dat ; done;

for i in 1 2 4 8 16 32; do echo $i ; mpirun -np $i  ./petscsol_rel.exe -nx 500 -ny 500 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre ; mv time_out.dat  500by500_time_out_$i.dat ; done;

for i in 1 2 4 8 16 32; do echo $i ; mpirun -np $i  ./petscsol_rel.exe -nx 1000 -ny 1000 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre ; mv time_out.dat  1000by1000_time_out_$i.dat ; done;

for i in 1 2 4 8 16 32; do echo $i ; mpirun -np $i  ./petscsol_rel.exe -nx 2000 -ny 2000 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre ; mv time_out.dat  2000by2000_time_out_$i.dat ; done;

for i in 1 2 4 8 16 32; do echo $i ; mpirun -np $i  ./petscsol_rel.exe -nx 3000 -ny 3000 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre ; mv time_out.dat  3000by3000_time_out_$i.dat ; done;

for i in 1 2 4 8 16 32; do echo $i ; mpirun -np $i  ./petscsol_rel.exe -nx 4000 -ny 4000 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre ; mv time_out.dat  4000by4000_time_out_$i.dat ; done;

for i in 1 2 4 8 16 32; do echo $i ; mpirun -np $i  ./petscsol_rel.exe -nx 5000 -ny 5000 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre ; mv time_out.dat  5000by5000_time_out_$i.dat ; done;

