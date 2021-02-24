#include <iostream>
#include <fstream>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
//#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Version.hpp>

#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <iostream>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp> 

#include <MatrixMarket_Tpetra.hpp>

#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosBiCGStabSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTFQMRSolMgr.hpp>
#include <BelosLSQRSolMgr.hpp>
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Ifpack2_ETIHelperMacros.h"
#include "Ifpack2_Details_Amesos2Wrapper.hpp"
#include "Ifpack2_Details_OneLevelFactory.hpp"
#include "Ifpack2_AdditiveSchwarz.hpp"
#include "Ifpack2_Factory.hpp"
#include "Ifpack2_Details_CanChangeMatrix.hpp"


#include <ctime>

using namespace Teuchos;
using namespace std; 

int main (int argc, char *argv[])
{

  typedef Tpetra::MultiVector<>::scalar_type scalar_type;
  typedef scalar_type ST;
  typedef Tpetra::MultiVector<>::local_ordinal_type  LO;
  typedef Tpetra::MultiVector<>::global_ordinal_type GO;
  typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
  
  typedef Teuchos::ScalarTraits<scalar_type> STS;
  typedef Tpetra::Map<LO, GO, node_type> map_type;
  typedef Tpetra::MultiVector<scalar_type, LO, GO, node_type> multivector_type;
  typedef Tpetra::CrsMatrix<scalar_type, LO, GO, node_type> sparse_mat_type;
  typedef Tpetra::Vector<>::scalar_type scalar_type;
  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
//  typedef MueLu::TpetraOperator<scalar_type,local_ordinal_type,global_ordinal_type,node_type> mtoperator;
  
  typedef Tpetra::Operator<scalar_type,local_ordinal_type,global_ordinal_type,node_type>    operator_type;
  typedef Belos::LinearProblem<scalar_type, multivector_type, operator_type> linear_problem_type;
  typedef Belos::SolverManager<scalar_type, multivector_type, operator_type> belos_solver_manager_type;

  typedef Belos::BlockGmresSolMgr<scalar_type, multivector_type, operator_type> belos_gmres_manager_type;
  typedef Belos::BiCGStabSolMgr<scalar_type, multivector_type, operator_type> belos_bicgstab_manager_type;
  typedef Belos::BlockCGSolMgr<scalar_type, multivector_type, operator_type>    belos_blockcg_manager_type;
  typedef Belos::PseudoBlockCGSolMgr<scalar_type, multivector_type, operator_type> belos_pseudocg_manager_type;
//  typedef Belos::BlockGmresSolMgr<scalar_type, multivector_type, operator_type> belos_gmres_manager_type;
//  typedef Belos::BiCGStabSolMgr<scalar_type, multivector_type, operator_type> belos_bicgstab_manager_type;
  typedef Belos::TFQMRSolMgr<scalar_type, multivector_type, operator_type>      belos_tfqmr_manager_type;
  typedef Belos::LSQRSolMgr<scalar_type, multivector_type, operator_type>       belos_lsqr_manager_type;

  typedef Ifpack2::Preconditioner<ST,LO,GO,node_type> prec_type; 
  std::cout << "Execution space name: ";
  typedef Tpetra::Map<>::device_type::execution_space default_execution_space;
  std::cout << Teuchos::TypeNameTraits<default_execution_space>::name () << std::endl;

  using Tpetra::global_size_t;
  using Teuchos::Array;
  using Teuchos::ArrayView;
  using Teuchos::ArrayRCP;
  using Teuchos::arcp;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  using Teuchos::parameterList;
//  using Teuchos::Time;
  using Teuchos::TimeMonitor;


  Teuchos::oblackholestream blackhole;
  using Teuchos::updateParametersFromXmlFile;
  using Teuchos::updateParametersFromXmlString; 


//  typedef Tpetra::CrsMatrix<> crs_matrix_type;
  Teuchos::GlobalMPISession mpiSession ();
  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
//  RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  const int myRank = comm->getRank ();
  const int numProcs = comm->getSize ();
 
  RCP<Time> insertva     = TimeMonitor::getNewCounter ("InsertValues ");
  RCP<Time> FillTimer    = TimeMonitor::getNewCounter ("FillComplete A");

  std::ostream &out = std::cout;
  RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));

  //long long int trows ; 

  std::string matrixFileName, vname ;

  ifstream myfile ("input.txt");
  if (myfile.is_open())
  {
//   myfile >>  trows ;
   myfile >> matrixFileName ; 
   myfile >> vname ; 
  }

  Teuchos::ParameterList defaultParameters;

  Teuchos::ParameterList params;
  using matrix_t = Tpetra::CrsMatrix<double>;
  using reader_t = Tpetra::MatrixMarket::Reader<matrix_t>;
//  Amat = reader_t::readSparseFile(filename, comm, params);
  Teuchos::RCP<sparse_mat_type> A = reader_t::readSparseFile(matrixFileName,comm, params); 
//  Teuchos::RCP<sparse_mat_type> A = Tpetra::MatrixMarket::Reader<sparse_mat_type>::readSparseFile(matrixFileName, comm, node_type); 
  //removed node

  std::clock_t c_start3 = std::clock();

  std::string prectype;
  myfile >>  prectype ; 

  std::string ifpack2par="ifpackpre.xml";
  Ifpack2::Factory factory;
  //DIAGONAL", "RELAXATION", "CHEBYSHEV", "ILUT", "RILUK"
  //RCP<prec_type> ifpack2Preconditioner = factory.create<crs_matrix_type> ("CHEBYSHEV", A);
  RCP<prec_type> ifpack2Preconditioner = factory.create<sparse_mat_type> (prectype, A);
  //ifpack2Preconditioner->setParameters( paramList );
  ifpack2Preconditioner->setParameters( ifpack2par);
  ifpack2Preconditioner->initialize();
  ifpack2Preconditioner->compute();
  
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

  std::clock_t c_start4 = std::clock();

  std::string xmlFileName = "test.xml";
  
  std::clock_t c_start5 = std::clock();
//   TimeMonitor monitor(*makeprb); 
  RCP<linear_problem_type> Problem = rcp(new linear_problem_type(A, x, b));
  Problem->setProblem();

  std::clock_t c_start6 = std::clock();

 //  TimeMonitor monitor(*SolsTimer) ;  
  RCP<Teuchos::ParameterList> belosList = rcp(new Teuchos::ParameterList()); 
// RCP<ParameterList> belosList = rcp(new ParameterList());
  belosList->set("Maximum Iterations",    200);    // Maximum number of iterations allowed
  belosList->set( "Num Blocks", 300);                // Maximum number of blocks in Krylov factorization
//  belosList->set("Maximum Restarts", 1000); 
  belosList->set("Convergence Tolerance", 1.0e-4);     // Relative convergence tolerance requested
  belosList->set("Block Size",          1);
  belosList->set("Verbosity",  Belos::Errors + Belos::Warnings + Belos::StatusTestDetails + Belos::TimingDetails + Belos::OrthoDetails + Belos::IterationDetails + Belos::Debug );
  belosList->set("Output Frequency",      20);
  belosList->set("Output Style",          Belos::Brief);
  belosList->set("Estimate Condition Number"  , true) ; 

  RCP<belos_solver_manager_type> solver;
  Problem->setRightPrec (ifpack2Preconditioner);

  if(vname=="GMRES") {
      solver = rcp(new belos_gmres_manager_type(Problem, belosList));
  }
  else if(vname=="CG")  {
      solver = rcp(new belos_blockcg_manager_type(Problem, belosList));
  }
  else if(vname=="TFQMR")     {
      solver = rcp(new belos_tfqmr_manager_type(Problem, belosList));
  }
  else if(vname=="BICG") {
      solver = rcp(new belos_bicgstab_manager_type(Problem, belosList));
  }
  else if(vname=="LSQR")     {
      solver = rcp(new  belos_lsqr_manager_type(Problem, belosList));
  }
  else
    {
      solver = rcp(new belos_bicgstab_manager_type(Problem, belosList));
  }

  solver->solve();
  std::clock_t c_start7 = std::clock();
  if (comm->getRank() == 0) {
     std::cout << "End Result" << std::endl;
  }

  return 0;

}
