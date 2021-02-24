
/* Author: VMK Kotteda
 * date  : Feb 18, 2021 */
#include <iostream>
#include <fstream>
#include "mmio.hpp"

//#include <cstdlib>
using namespace std;
#include <petscksp.h>
#include <stdio.h>
static char help[] = "Solves a linear system.\n\n";

int itersolve(int ne, int nn, int *row, int *col1,double *val)
{
/*
  cout << nn << std:: endl; 
  for (int i=0;  i < 100; i++) {
	cout <<i<<" "<<rhs[i]<<" "<<row[i]<<" "<<col1[i]<<" "<<val[i]<<std::endl; 

  }
*/
  Vec            x, b, u;          /* approx solution, RHS, exact solution */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = nn,col[3],its,rstart,rend,nlocal;
  PetscInt       start, end,rank,size;
  PetscInt       istart, iend;
  PetscScalar    value[3],vv;
  PetscRandom    rnd; 
  int            ii, jj; 

  //ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscInitialize(NULL, NULL, (char*)0, help); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  PetscRandomCreate(PETSC_COMM_SELF,&rnd);
  // get value from command line
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -rnd
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

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */

  ierr = VecGetOwnershipRange(x,&istart,&iend);CHKERRQ(ierr);
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
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
//  printf("%d \n",rank);
  ierr = MatGetOwnershipRange(A, &start,&end);

  /*
  This just takes any PetSC command line options that you might have supplied at run time and 
  apply them to the matrix, if you wanted to control how A was set up; 
  otherwise, you could just, have eg, used MatCreateMPIAIJ() to create it as an AIJ-format matrix (the default), 
  MatCreateMPIDense() if it was going to be a dense matrix.
  */

//  ierr = MatMPIAIJSetPreallocation(A,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);
//  ierr = MatSeqAIJSetPreallocation(A,5,PETSC_NULL);CHKERRQ(ierr);

  /*
     Assemble matrix.
     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */

//  PetscPrintf(PETSC_COMM_WORLD,"%d %d %d \n",rank, start, end);

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=0; i<ne; i++) {
    ii = row[i]-1; jj = col1[i]-1;
     if(ii <n && jj < n) {
       MatSetValues(A,1,&ii,1,&jj,&val[i],INSERT_VALUES);
    }
  }


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
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

//  PetscBarrier((PetscObject) A);
  /* create vectors */
//  MatCreateVecs(A,&x,&b);
//  VecDuplicate(x,&u);

  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
     PetscRandomGetValue(rnd, &vv); 
//     ierr = VecSetValues(b,1,&i,&rhs[i],INSERT_VALUES);CHKERRQ(ierr);
     ierr = VecSetValues(b,1,&i,&vv,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  MatMult(A, u, b);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Create the linear solver and set various options
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  KSPCreate(PETSC_COMM_WORLD,&ksp);

   /*
      Set operators. Here the matrix that defines the linear system
      also serves as the matrix that defines the preconditioner.
   */
  KSPSetOperators(ksp,A,A);

   /*
      Set linear solver defaults for this problem (optional).
      - By extracting the KSP and PC contexts from the KSP context,
        we can then directly call any KSP and PC routines to set
        various options.
      - The following four statements are optional; all of these
        parameters could alternatively be specified at runtime via
        KSPSetFromOptions();
   */
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCJACOBI);
  KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

   /*
     Set runtime options, e.g.,
         -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     KSPSetFromOptions() is called _after_ any other customization
     routines.
   */
  KSPSetFromOptions(ksp);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve the linear system
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  KSPSolve(ksp,b,x);

   /*
      View solver info; we could instead use the option -ksp_view to
      print this info to the screen at the conclusion of KSPSolve().
   */
  KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Check the solution and clean up
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecAXPY(x,-1.0,u);
  VecNorm(x,NORM_2,&norm);
  KSPGetIterationNumber(ksp,&its);
  PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);
 // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr); 
  ierr = MatViewFromOptions(A,NULL,"-assembled_view");CHKERRQ(ierr); 
  /*
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
   */
  VecDestroy(&x); VecDestroy(&u);
  VecDestroy(&b); MatDestroy(&A);
  KSPDestroy(&ksp);

   /*
      Always call PetscFinalize() before exiting a program.  This routine
        - finalizes the PETSc libraries as well as MPI
        - provides summary and diagnostic information if certain runtime
          options are chosen (e.g., -log_view).
   */
   PetscFinalize();
   return ierr;
}
