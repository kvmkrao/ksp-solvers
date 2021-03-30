
/* Author: VMK Kotteda
 * date  : Feb 18, 2021 */
#include <iostream>
#include <fstream>
//#include <cstdlib>
using namespace std;
#include <petscksp.h>
#include <stdio.h>
static char help[] = "Solves a linear system.\n\n";

/*
  cout << nn << std:: endl; 
  for (int i=0;  i < 100; i++) {
	cout <<i<<" "<<rhs[i]<<" "<<row[i]<<" "<<col1[i]<<" "<<val[i]<<std::endl; 

  }
*/
#include<ctime>
#include<iostream>
using namespace std; 
int main(int argc,char **args)
{
  Vec            x, b, u;          /* approx solution, RHS, exact solution */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,j,n=1515674,nx,ny,col[5],its,rstart,rend,nlocal;
  PetscInt       start, end,rank,size,irow,icol,ir,ic;
  PetscInt       istart, iend;
  PetscScalar    value[5],val1,rhs1;
  PetscScalar    one = 1.0; 
  PetscInt       numiter =10,nnz=0; 
//  PetscInt       d_nnz[n];
  int            ii,max=7219812; 
  string         itsol,prcd,word;

  ifstream infile;
  if(!infile) {
    cout << "Error in opening the input file" << std::endl;
  }
  infile >> nx;
  infile >> ny;
//  infile >> itsol ;
//  infile >> prcd ;
  infile.close();


  std::clock_t c_start0 = std::clock();
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
//  ierr = PetscInitialize(NULL, NULL, (char*)0, help); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute the matrix and right-hand-side vector that define
    the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
    Create vectors.  Note that we form 1 vector from scratch and
    then duplicate as needed. For this simple case let PETSc decide how
    many elements of the vector are stored on each processor. The second
    argument to VecSetSizes() below causes PETSc to decide.
  */
  // n - global size

  /*
  PetscInt   *d_nnz, *o_nnz; 
  ierr = PetscMalloc1(m+1, &d_nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(m+1, &o_nnz);CHKERRQ(ierr);
  

  ifstream nzfile;
  nzfile.open("nnz_mat.dat");
  if(!nzfile) {
    cout << "Error in opening the input file" << std::endl;
  }

  for (i=0; i<n; i++) {
      nzfile >> d_nnz[i] ;
//      cout <<  d_nnz[i] << endl; 
   }
//  return 0; 

  */

 
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);


  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */

//  ierr = VecGetOwnershipRange(x,&istart,&iend);CHKERRQ(ierr);
//  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.
     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
     We pass in nlocal as the "local" size of the matrix to force it
     to have the same parallel layout as the vector created above.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
//  ierr = MatSetSizes(A,nlocal,nlocal,n,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatMPIAIJSetPreallocation(A,65,NULL,65,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,65,NULL);CHKERRQ(ierr);
/*  
  ierr = MatSeqSBAIJSetPreallocation(A,1,d_nnz,NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,1,d_nnz,NULL,d_nnz,NULL);CHKERRQ(ierr);
  ierr = MatMPISELLSetPreallocation(A,d_nnz,NULL,d_nnz,NULL);CHKERRQ(ierr);
  ierr = MatSeqSELLSetPreallocation(A,5,NULL);CHKERRQ(ierr);
*/

  ierr = MatSetUp(A);CHKERRQ(ierr);
//  printf("%d \n",rank);
  ierr = MatGetOwnershipRange(A, &start,&end);
  /*
     Assemble matrix.
     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */

  printf("%d %d %d %d %d \n",rank, start, end, nx, ny, n);

  std::clock_t c_start1 = std::clock();
//  ifstream matfile;
//  matfile.open("sparsemat_S0.dat");
//  if(!matfile) {
//             cout << "Error in opening the file" << std::endl;
//  }
//  matfile >> word >> word >> word ;

 
  ifstream matfile;
  matfile.open("/home/vkotteda//utility/from_amin/5mNetwork/val_S0.dat");
  if(!matfile) {
             cout << "Error in opening the file" << std::endl;
  }
  matfile >> word >> word >> word ;

  for (i=0; i<max; i++) {
      matfile >> irow >> icol >> val1;
      ir = irow - 1; 
      ic = icol - 1; 
      if(ir >= start && ir < end) { //check the process own the row or not
        MatSetValues(A,1,&ir,1,&ic,&val1,INSERT_VALUES); //insert if it own
      }
  }
  matfile.close();

  /* each processor generates some of the matrix values */
  /*
  for (i=istart; i<iend; i++) {
     col[0] = i-1; col[1] = i; col[2] = i+1;
     if(i==0) {
	 MatSetValues(A,1,&i,2,col+1,value+1,INSERT_VALUES);
     }
     if(i== n-1) {
	 MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
     }
     else {
     MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);
     printf("%d %d %d %d %g %g\n",  rank, i, col[0], col[1],value[0],value[1]);
     }
  }
*/

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  std::clock_t c_start2 = std::clock();

  ierr = MatCreateVecs(A,&u,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  ierr = VecSet(u,1.0);CHKERRQ(ierr);
    std::clock_t c_start3 = std::clock();

  ierr = MatMult(A,u,b);CHKERRQ(ierr);

//  flg  = PETSC_FALSE;
//  ierr = PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL);CHKERRQ(ierr);
//  if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
 std::clock_t c_start4 = std::clock();

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
  if (test) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHMG);CHKERRQ(ierr);
    ierr = PCHMGSetInnerPCType(pc,PCGAMG);CHKERRQ(ierr);
    ierr = PCHMGSetReuseInterpolation(pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCHMGSetUseSubspaceCoarsening(pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCHMGUseMatMAIJ(pc,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PCHMGSetCoarseningComponent(pc,0);CHKERRQ(ierr);
  }
  */

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
   std::clock_t c_start5 = std::clock();

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  std::clock_t c_start6 = std::clock();

  /*
  if (reuse) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    // Make sparsity pattern different and reuse interpolation //
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
    n = 0;
    v = 0;
    m--;
    // Connect the last element to the first element //
    ierr = MatSetValue(A,m,n,v,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  }
*/  

  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its);CHKERRQ(ierr);


  ofstream outfile;
  outfile.open("time_out.dat");
  
  if(rank == 0) { 
     std::cout << "setup time "          << (double)(c_start1 - c_start0)/CLOCKS_PER_SEC << std::endl;
     std::cout << "fill matrix: time "   << (double)(c_start2 - c_start1)/CLOCKS_PER_SEC << std::endl;
     std::cout << "set up vectors: time "<< (double)(c_start3 - c_start2)/CLOCKS_PER_SEC << std::endl;
     std::cout << "setup ksp "           << (double)(c_start5 - c_start4)/CLOCKS_PER_SEC << std::endl;
     std::cout << "Solve "               << (double)(c_start6 - c_start5)/CLOCKS_PER_SEC << std::endl;
     std::cout << "Total "               << (double)(c_start6 - c_start0)/CLOCKS_PER_SEC << std::endl;
     if (outfile.is_open()) {
        outfile <<  "rank                " << size << " nx " << nx <<" ny " << ny          << std::endl;
        outfile << "setup time           " << (double)(c_start1 - c_start0)/CLOCKS_PER_SEC << std::endl;
        outfile << "fill matrix: time    " << (double)(c_start2 - c_start1)/CLOCKS_PER_SEC << std::endl;
        outfile << "set up vectors: time " << (double)(c_start3 - c_start2)/CLOCKS_PER_SEC << std::endl;
        outfile << "setup ksp            " << (double)(c_start5 - c_start4)/CLOCKS_PER_SEC << std::endl;
        outfile << "Solve                " << (double)(c_start6 - c_start5)/CLOCKS_PER_SEC << std::endl;
        outfile << "Total                " << (double)(c_start6 - c_start0)/CLOCKS_PER_SEC << std::endl;
     }
  }
  outfile.close(); 


  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


