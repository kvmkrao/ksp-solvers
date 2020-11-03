#include <iostream>
#include <fstream>

#include <Teuchos_TypeNameTraits.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp> 

#include <MatrixMarket_Tpetra.hpp>

#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_TpetraOperator.hpp>
//#include <Xpetra_TpetraVector.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_UseDefaultTypes.hpp>

#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include "BelosConfigDefs.hpp"

#include "Ifpack2_Factory.hpp"
#include "Ifpack2_ETIHelperMacros.h"
#include "Ifpack2_Details_Amesos2Wrapper.hpp"
#include "Ifpack2_Details_OneLevelFactory.hpp"
#include "Ifpack2_AdditiveSchwarz.hpp"

#include <ctime>

using namespace Teuchos;
using namespace std; 

int main (int argc, char *argv[])
{

  Teuchos::oblackholestream blackHole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackHole);
  RCP<const Teuchos::Comm<int> > comm =
  Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  typedef Tpetra::MultiVector<>::scalar_type ST;
 
  typedef Tpetra::Map<>::local_ordinal_type LO;
  typedef Tpetra::Map<>::global_ordinal_type GO;
  typedef Tpetra::Map<>::node_type node_type;

//  typedef Kokkos::Compat::KokkosCudaWrapperNode     node_type;
//  typedef Kokkos::Compat::KokkosOpenMPWrapperNode  node_type; 
//  typedef Kokkos::Compat::KokkosThreadsWrapperNode node_type;
//  typedef Kokkos::Compat::KokkosSerialWrapperNode  node_type;

   typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
  
  typedef Tpetra::Map<LO, GO, node_type> map_type;

  typedef Tpetra::MultiVector<ST, LO, GO, node_type> multivector_type;
  typedef Tpetra::CrsMatrix<ST, LO, GO, node_type> sparse_mat_type;

  typedef MueLu::TpetraOperator<ST,LO,GO,node_type> mtoperator;
  
  typedef Tpetra::Operator<ST,LO,GO,node_type>    operator_type;
  typedef Belos::LinearProblem<ST, multivector_type, operator_type> linear_problem_type;
  typedef Belos::SolverManager<ST, multivector_type, operator_type> belos_solver_manager_type;

  typedef Belos::BlockCGSolMgr<ST, multivector_type, operator_type>    belos_blockcg_manager_type;
  typedef Belos::PseudoBlockCGSolMgr<ST, multivector_type, operator_type> belos_pseudocg_manager_type;
// flexible 
  typedef Belos::BlockGmresSolMgr<ST, multivector_type, operator_type>       belos_stdgmres_manager_type;
// standard 
  typedef Belos::PseudoBlockGmresSolMgr<ST, multivector_type, operator_type> belos_flexgmres_manager_type;

  typedef Belos::BiCGStabSolMgr<ST, multivector_type, operator_type> belos_bicgstab_manager_type;
  typedef Belos::TFQMRSolMgr<ST, multivector_type, operator_type>      belos_tfqmr_manager_type;
  
  typedef Belos::LSQRSolMgr<ST, multivector_type, operator_type>       belos_lsqr_manager_type;
  typedef Belos::GCRODRSolMgr<ST, multivector_type, operator_type>     belos_recgmres_type; 
  typedef Belos::GmresPolySolMgr<ST, multivector_type, operator_type>  belos_hybridgmres_type; 
  typedef Belos::PseudoBlockTFQMRSolMgr<ST, multivector_type, operator_type> belos_psedotfqmr_type;  
//  typedef Belos::PseudoBlockCGSolMgr<ST, multivector_type, operator_type> belos_psedobcg_type;  

  typedef Ifpack2::Preconditioner<ST,LO,GO,node_type> prec_type; 

  std::cout << "Execution space name: ";
  typedef Tpetra::Map<>::device_type::execution_space default_execution_space;
  std::cout << Teuchos::TypeNameTraits<default_execution_space>::name () << std::endl;

  using Tpetra::global_size_t;
//  using Teuchos::Array;
//  using Teuchos::ArrayView;
  using Teuchos::ArrayRCP;
  using Teuchos::arcp;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  using Teuchos::parameterList;
//  using Teuchos::Time;
  using Teuchos::TimeMonitor;

  const int myRank = comm->getRank ();
  const int numProcs = comm->getSize ();
 
  RCP<Time> insertva     = TimeMonitor::getNewCounter ("InsertValues ");
  RCP<Time> FillTimer    = TimeMonitor::getNewCounter ("FillComplete A");

  std::ostream &out = std::cout;
  RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));

// const global_size_t numGlobalElements = 200000000; 
   long long int trows ; 

  std::clock_t c_start0 = std::clock();

   std::string vname;

   std::string matrixFileName ;
  ifstream myfile ("input.txt");
   if (myfile.is_open())
  {

   myfile >>   vname ;
   //myfile >>  trows ;
   myfile >> matrixFileName ; 
  }

  std::cout << "matrix market name " << matrixFileName << std::endl ; 

  Teuchos::ParameterList defaultParameters;
  Teuchos::RCP <node_type> node = Teuchos::rcp(new node_type(defaultParameters));

  Teuchos::RCP<sparse_mat_type> A = Tpetra::MatrixMarket::Reader<sparse_mat_type>::readSparseFile(matrixFileName, comm, node); //removed node

  std::clock_t c_start1 = std::clock();

 //Note:: Tpetra::MatrixMarket::Reader is providing A->getColMap() wrong and equal A->row
   Teuchos::RCP<multivector_type> x = Teuchos::rcp(new multivector_type(A->getRowMap(), 1));
   Teuchos::RCP<multivector_type> b = Teuchos::rcp(new multivector_type(A->getRowMap(), 1));
   b->randomize();
   x->randomize();
  cout << "num_vector: " << b->getNumVectors() << " "
  << x->getNumVectors() << endl;
  cout << "length: " << b->getGlobalLength() << " "
  << x->getGlobalLength() << endl;
  cout << "A length" << A->getGlobalNumRows() << " " << A->getGlobalNumCols() << endl;
  cout << "A local length" << A->getNodeNumRows() << " " << A->getNodeNumCols() << endl;

   std::clock_t c_start2 = std::clock();

  std::string xmlFileName = "setprec.xml";
//   TimeMonitor monitor(*makeprb); 
   RCP<linear_problem_type> Problem = rcp(new linear_problem_type(A, x, b));
   Problem->setProblem();

  std::clock_t c_start3 = std::clock();

 //  TimeMonitor monitor(*SolsTimer) ;   
  RCP<ParameterList> belosList = rcp(new ParameterList());
  belosList->set("Maximum Iterations",    20);    // Maximum number of iterations allowed
  belosList->set( "Num Blocks", 300);                // Maximum number of blocks in Krylov factorization
//  belosList->set("Maximum Restarts", 1000); 
  belosList->set("Convergence Tolerance", 1.0e-4);     // Relative convergence tolerance requested
  belosList->set("Block Size",          1);
  belosList->set("Verbosity",  Belos::Errors + Belos::Warnings + Belos::StatusTestDetails + Belos::TimingDetails + Belos::OrthoDetails + Belos::IterationDetails + Belos::Debug );
  belosList->set("Output Frequency",      20);
  belosList->set("Output Style",          Belos::Brief);
  belosList->set("Estimate Condition Number"  , true) ; 

  RCP<belos_solver_manager_type> solver;

  std::clock_t c_start4 = std::clock();

  std::string prectype,precName;
  myfile >>  prectype ;
  myfile >>  precName ;

  if(precName == "ifpack2") {
    std::string ifpack2par="ifpackpre.xml";
    Ifpack2::Factory factory;
    //DIAGONAL", "RELAXATION", "CHEBYSHEV", "ILUT", "RILUK"
    //RCP<prec_type> ifpack2Preconditioner = factory.create<crs_matrix_type> ("CHEBYSHEV", A);
    RCP<prec_type> ifpack2Preconditioner = factory.create<sparse_mat_type> (prectype, A);
    //ifpack2Preconditioner->setParameters( paramList );
    ifpack2Preconditioner->setParameters( ifpack2par);
    ifpack2Preconditioner->initialize();
    ifpack2Preconditioner->compute();
    Problem->setRightPrec (ifpack2Preconditioner);
    }
   else 
   {
    RCP< mtoperator > mueLuPreconditioner = MueLu::CreateTpetraPreconditioner<ST,LO,GO,node_type>(RCP<operator_type>(A), xmlFileName);
    Problem->setRightPrec (mueLuPreconditioner);
  }


    std::clock_t c_start5 = std::clock();

    if(vname=="GMRES") {
     solver = rcp(new belos_stdgmres_manager_type(Problem, belosList));
    }
    else if(vname=="CG")  {
     solver = rcp(new belos_blockcg_manager_type(Problem, belosList));
    }
    else if(vname=="FlexGMRES") {
     solver = rcp(new belos_flexgmres_manager_type(Problem, belosList));
    }
    else if(vname=="TFQMR")     {
     solver = rcp(new belos_tfqmr_manager_type(Problem, belosList));
    }
    else if(vname=="RecyleGMRES") {
     solver = rcp(new belos_recgmres_type (Problem, belosList));
    }

    else if (vname=="BiCGStab") {
     solver = rcp(new belos_bicgstab_manager_type(Problem, belosList));
    }
    else {
     solver = rcp(new belos_blockcg_manager_type(Problem, belosList));
    }

  std::clock_t c_start6 = std::clock();
  solver->solve();
  std::clock_t c_start7 = std::clock();

  if(myRank ==0 ) {
    std::cout << "read matrix "<< 1000*(c_start1-c_start0)/CLOCKS_PER_SEC   << std::endl;
    std::cout << "make x & b "<< 1000*(c_start2-c_start1)/CLOCKS_PER_SEC   << std::endl;
    std::cout << "Ax = b "    << 1000*(c_start3-c_start2)/CLOCKS_PER_SEC   << std::endl;
    std::cout << "set solver parameters"<< 1000*(c_start4-c_start3)/CLOCKS_PER_SEC   << std::endl;
    std::cout << "setup preconditioner" << 1000*(c_start5-c_start4)/CLOCKS_PER_SEC   << std::endl;
    std::cout << "set solver   "   << 1000*(c_start6-c_start5)/CLOCKS_PER_SEC   << std::endl;
    std::cout << "Solver       "   << 1000*(c_start7-c_start6)/CLOCKS_PER_SEC   << std::endl;
    std::cout << "total time "      << 1000*(c_start7-c_start0)/CLOCKS_PER_SEC   << std::endl;
    int numIterations = solver->getNumIters();
    std::cout << " number of iterations" <<  numIterations << std::endl;
    double tolach = solver->achievedTol( );
    std::cout << " tolerance achieved" <<  tolach << std::endl;
  }



}
