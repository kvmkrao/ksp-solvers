
/* Author: VMK Kotteda
 * date  : Feb 24, 2021 
 * mpirun -np 32  ./petscsol_rel.exe  -nx 1000 -ny 1000 -nz 5 -ksp_atol  1e-06   -ksp_rtol  1e-10 -ksp_type cg -pc_type hypre   -ksp_monitor_short  */

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
  PetscInt       i,j,n,nx,ny,nz,col[7],its,rstart,rend,nlocal;
  PetscInt       start, end,rank,size,irow,icol;
  PetscInt       istart, iend;
  PetscScalar    value[7],val1,rhs1;
  PetscScalar    one = 1.0; 
  PetscInt       numiter =10,nnz=0; 
//  PetscInt       d_nnz[n];
  int            ii,max; 
  string         itsol,prcd,word;

  ifstream infile;
  infile.open("input.in");
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

  ierr = PetscOptionsGetInt(NULL,NULL,"-nx",&nx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ny",&ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nz",&nz,NULL);CHKERRQ(ierr);

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

/*  
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
*/

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
   n = nx * ny* nz; 
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
//  ierr = MatSetSizes(A,nlocal,nlocal,n,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatMPIAIJSetPreallocation(A,7,NULL,7,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,7,NULL);CHKERRQ(ierr);
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

  printf("%d %d %d %d %d %d\n",rank, start, end, nx, ny, nz, n);

  std::clock_t c_start1 = std::clock();
//  ifstream matfile;
//  matfile.open("sparsemat_S0.dat");
//  if(!matfile) {
//             cout << "Error in opening the file" << std::endl;
//  }
//  matfile >> word >> word >> word ;

  

  for (i=start; i<end; i++) {
    nnz        = 0;
    col[nnz]   = i; 
    value[nnz] = 4.0; 

    if((i-nz)>=0) {
      nnz        = nnz + 1;
      col[nnz]   = i-nz;
      value[nnz] = -1.0;
    }


    if((i-ny)>=0) {
      nnz        = nnz + 1; 
      col[nnz]   = i-ny;   
      value[nnz] = -1.0;
    }

    if((i-1)>=0 ) {
      nnz        = nnz+1 ;
      col[nnz]   = i-1;
      value[nnz] = -1.0;
    }
    
    if((i+1)<=n-1 ) {
      nnz        = nnz+1 ;
      col[nnz]   = i+1;
      value[nnz] = -1.0;
    }

    if((i+ny)<=n-1) {
      nnz        = nnz + 1;
      col[nnz]   = i+ny;
      value[nnz] = -1.0;
    }
    
    if((i+nz)<=n-1) {
      nnz        = nnz + 1;
      col[nnz]   = i+nz;
      value[nnz] = -1.0;
    }

    ierr = MatSetValues(A,1,&i,nnz+1,col,value,INSERT_VALUES);CHKERRQ(ierr);
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
 
    /*
       Indicate same nonzero structure of successive linear system matrices
    */
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE);CHKERRQ(ierr);
  std::clock_t c_start2 = std::clock();
//  PetscBarrier((PetscObject) A);
  /* create vectors */
//  MatCreateVecs(A,&x,&b);
//  VecDuplicate(x,&u);

  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
//  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
//  ierr = VecSet(u,1.0);CHKERRQ(ierr);


/*  
  ifstream rfile,xfile;
  rfile.open("rhs_S0.dat");
  xfile.open("indx_initp_S0.dat");

    // read right hand side vector 
  if(!rfile) {
             cout << "Error in opening the file" << std::endl;
   }

   rfile >> word >> word ;
  
   for(int i=0; i< n; i++) {
      rfile >> ii >> rhs[i] ;
//      cout << ii << " " << rhs[i] << std::endl;
   }
   rfile.close();
  

   if(!xfile) {
      cout << "Error in opening the file" << std::endl;
   }
   */

//   xfile >> word ;
/*   for(int i=0; i< nodes; i++) {
      xfile >> xi[i] ;
      cout << i << " " << rhs[i] << std::endl;
   }
*/


/*
  //xfile >> word ;
    for (i=0; i<n; i++) {
     rfile >> ii >> rhs1 ;
     if(ii >= start && ii < end) { //check if the processor own
       ierr = VecSetValues(b,1,&ii,&rhs1,INSERT_VALUES);CHKERRQ(ierr);
     }
  }
  
  //VecSet(b,one); 
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
*/


  std::clock_t c_start3 = std::clock();
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
    std::clock_t c_start4 = std::clock();
    /* Solve the linear system */
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    std::clock_t c_start5 = std::clock();


  for(int j=1; j<3; j++) { 
    std::clock_t c_start5 = std::clock();
    // resume fill

    /*
     Initialize all matrix entries to zero.  MatZeroEntries() retains the
     nonzero structure of the matrix for sparse formats.
    */
    ierr = MatZeroEntries(A);CHKERRQ(ierr);

    // allow making changes to memory size and new non-zero values 

    if(j==1) {
      for(i=start; i<end; i++) {
        nnz = 0;
        col[nnz]   = i;
        value[nnz] = 4.0;
  
        if((i-ny)>=0) {
          nnz = nnz + 1;
          col[nnz]   = i-ny;
          value[nnz] = -1.0;
        }
  
        if((i-1)>=0 ) {
          nnz = nnz+1 ;
          col[nnz]   = i-1;
          value[nnz] = -1.0;
        }
  
        if((i+1)<=n-1 ) {
          nnz = nnz+1 ;
          col[nnz]   = i+1;
          value[nnz] = -1.0;
        }
  
        if((i+ny)<=n-1) {
          nnz = nnz + 1;
          col[nnz]   = i+ny;
          value[nnz] = -1.0;
        }
  
        ierr = MatSetValues(A,1,&i,nnz+1,col,value,INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    else if (j==2) {
      for(i=start; i<end; i++) {
        nnz = 0;   
        col[nnz]   = i;
        value[nnz] = 4.0;

        if((i-1)>=0 ) {
          nnz = nnz+1 ;
          col[nnz]   = i-1;
          value[nnz] = -1.0;
        }

        if((i+1)<=n-1 ) {
          nnz = nnz+1 ;
          col[nnz]   = i+1;
          value[nnz] = -1.0;
        }
        ierr = MatSetValues(A,1,&i,nnz+1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
    }

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    std::clock_t c_start6 = std::clock();
    // Update b //
    ierr = MatMult(A,x,b);CHKERRQ(ierr);

    // Solve the linear system //
    std::clock_t c_start7 = std::clock();
    /*
       Use the previous solution of linear system #1 as the initial
       guess for the next solve of linear system #1.  The user MUST
       call KSPSetInitialGuessNonzero() in indicate use of an initial
       guess vector; otherwise, an initial guess of zero is used.
    */
      ierr = KSPSetInitialGuessNonzero(ksp,PETSC_FLASE);CHKERRQ(ierr);


    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    std::clock_t c_start8 = std::clock();
    if(rank==0) std::cout << "loop: j " << j <<"fill matrix "<< (double)(c_start6 - c_start5)/CLOCKS_PER_SEC << std::endl;
    if(rank==0) std::cout << "loop: j " << j <<"linear solver "<< (double)(c_start8 - c_start7)/CLOCKS_PER_SEC << std::endl;
    // Check the solution and clean up //
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
 //   if (norm > 100*PETSC_MACHINE_EPSILON) {
    if(rank==0) PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);
//    }
  }

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve the linear system
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   //ksp - iterative context obtained from KSPCreate()
   //b	 - the right hand side vector
   //x	 - the solution (this may be the same vector as b, then b will be overwritten with answer)

  /*
  std::clock_t c_start3 = std::clock();
  KSPSolve(ksp,b,x);
  std::clock_t c_start4 = std::clock();
  */
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
  ofstream outfile;
  outfile.open("time_out.dat");
  
  if(rank == 0) { 
     std::cout << "setup time "          << (double)(c_start1 - c_start0)/CLOCKS_PER_SEC << std::endl;
     std::cout << "fill matrix: time "   << (double)(c_start2 - c_start1)/CLOCKS_PER_SEC << std::endl;
     std::cout << "set up vectors: time "<< (double)(c_start3 - c_start2)/CLOCKS_PER_SEC << std::endl;
     std::cout << "setup ksp "           << (double)(c_start4 - c_start3)/CLOCKS_PER_SEC << std::endl;
     std::cout << "Solve "               << (double)(c_start5 - c_start4)/CLOCKS_PER_SEC << std::endl;
     std::cout << "Total "               << (double)(c_start5 - c_start0)/CLOCKS_PER_SEC << std::endl;
     if (outfile.is_open()) {
         outfile <<  "rank   "              << size << " nx " << nx <<" ny " << ny          << std::endl;
         outfile <<  "setup time          " << (double)(c_start1 - c_start0)/CLOCKS_PER_SEC << std::endl;
         outfile <<  "fill matrix: time   " << (double)(c_start2 - c_start1)/CLOCKS_PER_SEC << std::endl;
         outfile <<  "set up vectors:time " << (double)(c_start3 - c_start2)/CLOCKS_PER_SEC << std::endl;
         outfile <<  "setup ksp           " << (double)(c_start4 - c_start3)/CLOCKS_PER_SEC << std::endl;
         outfile <<  "Solve               " << (double)(c_start5 - c_start4)/CLOCKS_PER_SEC << std::endl;
         outfile <<  "Total               " << (double)(c_start5 - c_start0)/CLOCKS_PER_SEC << std::endl;
     }
  }
  outfile.close(); 

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Check the solution and clean up
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  //VecAXPY(x,-1.0,u);
  VecNorm(x,NORM_2,&norm);
//  KSPGetIterationNumber(ksp,&its);
//  PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);

 // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr); 
  ierr = MatViewFromOptions(A,NULL,"-assembled_view");CHKERRQ(ierr); 
  /*
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
   */
  VecDestroy(&x); 
  //VecDestroy(&u);
  VecDestroy(&b); MatDestroy(&A);
  KSPDestroy(&ksp);

   /*
      Always call PetscFinalize() before exiting a program.  This routine
        - finalizes the PETSc libraries as well as MPI
        - provides summary and diagnostic information if certain runtime
          options are chosen (e.g., -log_view).
   */
  ierr = PetscFinalize();
  return ierr;
}


