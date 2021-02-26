

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

#include <fstream>
#include <iostream>
#include <petscksp.h>
//#include <petscmat.h>
using namespace std;
int main(int argc,char **args)
{
  Vec            x, b, u;          /* approx solution, RHS, exact solution */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;             // preconditioner context 
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,m,n,col[3],its,rstart,rend,nlocal,ii;
  PetscInt       start, end,rank,size, pst,ped,np; 
  PetscInt       istart, iend,bs; 
  PetscScalar    value[3];
  string         itsol,prcd;
  PetscInt       *dnnz,*onnz; 

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  cout << rank << " "<< size <<endl;  

  ifstream infile;
  infile.open("input.dat");
  if(!infile) {
    cout << "Error in opening the input file" << std::endl;
  }
  infile >> n; 
  infile >> itsol ;
  infile >> prcd ; 
  infile.close(); 
  m = n;  
  bs= 3;

//  cout << rank << " "<< size <<" "<<n<<" "<<np <<endl;
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

  /*
   ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
   ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  */
 // ierr = MatCreateAIJ(PETSC_COMM_WORLD,n,n,3,NULL,&A);CHKERRQ(ierr);
  
  /*
  PetscMalloc2(n,&dnnz,n,&onnz);
  for (i=0; i<n; i++) {
    dnnz[i] = 1;  //array containing the number of nonzeros in the various rows of the DIAGONAL portion of the local submatrix
    onnz[i] = 1;  //array containing the number of nonzeros in the various rows of the OFF-DIAGONAL portion of the local submatrix
  }
  MatCreateAIJ(PETSC_COMM_WORLD,m,n,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_DECIDE,dnnz,PETSC_DECIDE,onnz,&A);
//  ierr = MatCreateAIJ(PETSC_COMM_WORLD,n,n,3,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  PetscFree2(dnnz,onnz);
  printf("%d \n",rank);
//  ierr = MatSetFromOptions(A);CHKERRQ(ierr);                                             
//  ierr = MatSetUp(A);CHKERRQ(ierr);      
  ierr = MatGetOwnershipRange(A, &start,&end);
*/

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,3,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&start,&end);CHKERRQ(ierr);
  /*
     Assemble matrix.

     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */

  PetscPrintf(PETSC_COMM_WORLD,"%d %d %d \n",rank, start, end);

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  cout << rank <<" "<< size <<" "<< start <<" "<<end << " "<<n << endl; 

  for (i=start; i<end; i++) {
    if(i==0) {
      col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if(i==n-1)  {
      col[0] = i; col[1] = i-1; value[0] = 2.0; value[1] = -1.0; 
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    else { 
    col[0] = i-1; col[1] = i; col[2] = i+1;
    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  }

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


  /*
     Assemble matrix.

     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  /* create vectors */
//  MatCreateVecs(A,&x,&b);
//  VecDuplicate(x,&u);

  VecSet(x,0.5); 
  VecSet(b,1.0); 

  MatMult(A, u, b); 
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Create the linear solver and set various options
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  KSPCreate(PETSC_COMM_WORLD,&ksp);
   /*
      Set operators. Here the matrix that defines the linear system
      also serves as the matrix that defines the preconditioner.
   */
  //KSPSetOperators(ksp,A,A);

   /*
      Set linear solver defaults for this problem (optional).
      - By extracting the KSP and PC contexts from the KSP context,
        we can then directly call any KSP and PC routines to set
        various options.
      - The following four statements are optional; all of these
        parameters could alternatively be specified at runtime via
        KSPSetFromOptions();
   */
  // set solver 
   ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
   if(itsol == "GMRES")  KSPSetType(ksp,KSPGMRES);
   if(itsol == "CG")     KSPSetType(ksp,KSPCG);
   if(itsol == "BICGS")  KSPSetType(ksp,KSPBICG);
   if(itsol == "BICGS")  KSPSetType(ksp,KSPBCGS);
//  set preconditioner  
    KSPGetPC(ksp,&pc);
    if(prcd == "JACOBI") ierr = PCSetType(pc,PCJACOBI); CHKERRQ(ierr);
    if(prcd == "SOR")    ierr = PCSetType(pc,PCSOR);CHKERRQ(ierr);
    if(prcd == "NONE")   ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    if(prcd == "ILU")    ierr = PCSetType(pc,PCILU);CHKERRQ(ierr); 
    KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
  // set the preconditioning side 
  //  ierr = KSPSetPCSide(ksp,PC_RIGHT); CHKERRQ(ierr);
  //  ierr = KSPSetNormType(ksp, KSP_NORM_NONE); CHKERRQ(ierr); 
     KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

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
//  VecAXPY(x,-1.0,u);
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
