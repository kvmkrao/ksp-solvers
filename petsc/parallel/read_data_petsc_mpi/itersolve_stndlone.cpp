
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
  PetscInt       i,n = 288420,col[3],its,rstart,rend,nlocal;
  PetscInt       start, end,rank,size,irow,icol;
  PetscInt       istart, iend;
  PetscScalar    value[3],val1,rhs1;
  int            ii,max=1368756;
  string         itsol,prcd,word;

  std::clock_t c_start0 = std::clock();
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
//  ierr = PetscInitialize(NULL, NULL, (char*)0, help); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  // get value from command line
  /*
  ifstream infile;
  infile.open("input.dat");
  if(!infile) {
    cout << "Error in opening the input file" << std::endl;
  }
  infile >> n; 
  infile >> itsol ;
  infile >> prcd ; 
  infile.close(); 
*/
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

//  PetscPrintf(PETSC_COMM_WORLD,"%d %d %d \n",rank, start, end);

  //value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  std::clock_t c_start1 = std::clock();
  ifstream matfile;
  matfile.open("sparsemat_S0.dat");
  if(!matfile) {
             cout << "Error in opening the file" << std::endl;
  }
  matfile >> word >> word >> word ;

  for (i=0; i<max; i++) { 
      matfile >> irow >> icol >> val1;
      if(irow >= start && irow < end) { //check the process own the row or not 
        MatSetValues(A,1,&irow,1,&icol,&val1,INSERT_VALUES); //insert if it own 
      }
  }
  matfile.close();
  std::clock_t c_start2 = std::clock();

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
  ifstream rfile,xfile;
  rfile.open("rhs_S0.dat");
  xfile.open("indx_initp_S0.dat");

    // read right hand side vector 
  if(!rfile) {
             cout << "Error in opening the file" << std::endl;
   }

   rfile >> word >> word ;
   /*
   for(int i=0; i< nodes; i++) {
      rfile >> ii >> rhs[i] ;
//      cout << ii << " " << rhs[i] << std::endl;
   }
   rfile.close();
   */

   if(!xfile) {
      cout << "Error in opening the file" << std::endl;
   }

//   xfile >> word ;
/*   for(int i=0; i< nodes; i++) {
      xfile >> xi[i] ;
      cout << i << " " << rhs[i] << std::endl;
   }
*/

  //xfile >> word ;
  for (i=0; i<n; i++) {
     rfile >> ii >> rhs1 ;
     if(ii >= start && ii < end) { //check if the processor own
       ierr = VecSetValues(b,1,&ii,&rhs1,INSERT_VALUES);CHKERRQ(ierr);
     }
  }
  rfile.close();

  for (i=0; i<n; i++) {
     xfile >> ii >> val1;
     if(ii >= start && ii < end) { //check if the processor own
       ierr = VecSetValues(x,1,&ii,&val1,INSERT_VALUES);CHKERRQ(ierr);
     }
  }
  xfile.close();

  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  //Computes the matrix-vector product, b = Ax. 
  MatMult(A, x, b);

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
 
  //KSPGetPC(ksp,&pc);
  //PCSetType(pc,PCJACOBI);
  KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  
    // set solver 
//   PetscStrcpy(itsolv,&itsol); 
  /*
   if(itsol == "GMRES")  KSPSetType(ksp,KSPGMRES);
   if(itsol == "CG")     KSPSetType(ksp,KSPCG);
   if(itsol == "BICGS")  KSPSetType(ksp,KSPBICG);
   if(itsol == "BICGS")  KSPSetType(ksp,KSPBCGS);
//   PetscStrcpy(prec,&prcd);
//  set preconditioner  
    KSPGetPC(ksp,&pc);
    if(prcd == "JACOBI") ierr = PCSetType(pc,PCJACOBI); CHKERRQ(ierr);
    if(prcd == "SOR")    ierr = PCSetType(pc,PCSOR);CHKERRQ(ierr);
    if(prcd == "NONE")   ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    if(prcd == "ILU")    ierr = PCSetType(pc,PCILU);CHKERRQ(ierr); 
    KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
  // set the preconditioning side 
    //ierr = KSPSetPCSide(ksp,PC_RIGHT); CHKERRQ(ierr);
    //ierr = KSPSetNormType(ksp, KSP_NORM_NONE); CHKERRQ(ierr); 
    KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
   */
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
   //ksp - iterative context obtained from KSPCreate()
   //b	 - the right hand side vector
   //x	 - the solution (this may be the same vector as b, then b will be overwritten with answer)
  std::clock_t c_start3 = std::clock();
  KSPSolve(ksp,b,x);
  std::clock_t c_start4 = std::clock();

   /*
      View solver info; we could instead use the option -ksp_view to
      print this info to the screen at the conclusion of KSPSolve().
   */
  KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);

  PetscViewer viewer;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "sol.dat", &viewer);
  //ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"solution.petsc",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(x, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  // print timings 
  //
  if(rank == 0) { 
     std::cout << "setup time "          << 1000*(c_start1 - c_start0)/CLOCKS_PER_SEC << std::endl;
     std::cout << "fill matrix: time "   << 1000*(c_start2 - c_start1)/CLOCKS_PER_SEC << std::endl;
     std::cout << "set up vectors: time "<< 1000*(c_start3 - c_start2)/CLOCKS_PER_SEC << std::endl;
     std::cout << "linear solve time "   << 1000*(c_start4 - c_start3)/CLOCKS_PER_SEC << std::endl;
     std::cout << "Total "               << 1000*(c_start4 - c_start0)/CLOCKS_PER_SEC << std::endl;
  }
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


