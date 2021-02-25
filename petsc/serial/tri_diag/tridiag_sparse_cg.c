

static char help[] = "Solves a tridiagonal linear system.\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Processors: n
T*/



/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

  Note:  The corresponding uniprocessor example is ex1.c
*/
#include <petscksp.h>
#include <stdio.h> 
int main(int argc,char **args)
{
  Vec            x, b, u;          /* approx solution, RHS, exact solution */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = 10000,col[3],its,rstart,rend,nlocal;
  PetscInt       start, end,rank,size; 
  PetscInt       istart, iend; 
  PetscScalar    value[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  // get value from command line  
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

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
  printf("%d \n",rank); 
  ierr = MatGetOwnershipRange(A, &start,&end);
  /*
     Assemble matrix.

     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */

  PetscPrintf(PETSC_COMM_WORLD,"%d %d %d \n",rank, start, end);
//  istart = start; if (start == 0) istart = 1;
//  iend = end;     if (end == n-1) iend = n-2;


  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

/*
  for (i=istart; i<iend; i++) { // each processor generates some of the matrix values 
     col[0] = i-1; col[1] = i; col[2] = i+1;
     if(i==0) {
	 MatSetValues(A,1,&i,2,col+1,value+1,INSERT_VALUES);
     }
     if(i== n-1) {
	 MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
     }
     else {
     MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);
     //printf("%d %d %d %d %g %g\n",  rank, i, col[0], col[1],value[0],value[1]);
     }
  }
*/

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

//  PetscBarrier((PetscObject) A); 
//  ierr = PetscFinalize(); 
  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */

  /* create vectors */
//  MatCreateVecs(A,&x,&b);
//  VecDuplicate(x,&u);

  VecSet(u,1.0); 
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



/*TEST

   build:
      requires: !complex !single
   test: 
     args: -ksp_type cg -ksp_monitor 

   test:
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      nsize: 3
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3
      nsize: 2
      args: -ksp_monitor_short -ksp_rtol 1e-6 -ksp_type pipefgmres

TEST*/
