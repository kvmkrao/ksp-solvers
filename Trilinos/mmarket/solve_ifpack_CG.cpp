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
//#include <MueLu.hpp>
//#include <MueLu_Exceptions.hpp>

//#include <MueLu_Hierarchy.hpp>
//#include <MueLu_MasterList.hpp>
//#include <MueLu_Utilities.hpp>

//  #include "MueLu_FactoryManagerBase.hpp"
//#include "MueLu_PFactory.hpp"
//#include "MueLu_SaPFactory.hpp"
//#include "MueLu_TransPFactory.hpp"
//#include "MueLu_RAPFactory.hpp"
//#include "MueLu_AmesosSmoother.hpp"
//#include "MueLu_TrilinosSmoother.hpp"
  //#include "MueLu_SmootherFactory.hpp"
  //#include "MueLu_CoupledAggregationFactory.hpp"
//#include "MueLu_TentativePFactory.hpp"
//#include "MueLu_AmesosSmoother.hpp"

//#include <MueLu_Level.hpp>

/*
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_TpetraOperator.hpp>
//#include <Xpetra_TpetraVector.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_UseDefaultTypes.hpp>
*/

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
/*
  typedef Tpetra::MultiVector<>::local_ordinal_type  LO;
  typedef Tpetra::MultiVector<>::global_ordinal_type GO;
  typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
*/
 
  typedef Tpetra::Map<>::local_ordinal_type LO;
  typedef Tpetra::Map<>::global_ordinal_type GO;
  typedef Tpetra::Map<>::node_type node_type;

//  typedef Kokkos::Compat::KokkosCudaWrapperNode     node_type;
//  typedef Kokkos::Compat::KokkosOpenMPWrapperNode  node_type; 
//  typedef Kokkos::Compat::KokkosThreadsWrapperNode node_type;
//  typedef Kokkos::Compat::KokkosSerialWrapperNode  node_type;

//typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
  
   typedef Tpetra::Details::DefaultTypes::node_type node_type;

  typedef Tpetra::Map<LO, GO, node_type> map_type;

  typedef Tpetra::MultiVector<ST, LO, GO, node_type> multivector_type;
  typedef Tpetra::CrsMatrix<ST, LO, GO, node_type> sparse_mat_type;

//  typedef Tpetra::Map<>::node_type NT;

/*
  typedef MueLu::TpetraOperator<ST,LO,GO,node_type> mtoperator;
*/  
  typedef Tpetra::Operator<ST,LO,GO,node_type>    operator_type;
  typedef Belos::LinearProblem<ST, multivector_type, operator_type> linear_problem_type;
  typedef Belos::SolverManager<ST, multivector_type, operator_type> belos_solver_manager_type;

  typedef Belos::BlockCGSolMgr<ST, multivector_type, operator_type>    belos_blockcg_manager_type;
  typedef Belos::PseudoBlockCGSolMgr<ST, multivector_type, operator_type> belos_pseudocg_manager_type;
// flexible 
  typedef Belos::BlockGmresSolMgr<ST, multivector_type, operator_type>       belos_flexgmres_manager_type;
// standard 
  typedef Belos::PseudoBlockGmresSolMgr<ST, multivector_type, operator_type> belos_stdgmres_manager_type;

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



/*  using Teuchos::updateParametersFromXmlFile;
  using Teuchos::updateParametersFromXmlString; 
*/

//  typedef Tpetra::CrsMatrix<> crs_matrix_type;

  const int myRank = comm->getRank ();
  const int numProcs = comm->getSize ();
 
  RCP<Time> insertva     = TimeMonitor::getNewCounter ("InsertValues ");
  RCP<Time> FillTimer    = TimeMonitor::getNewCounter ("FillComplete A");

  std::ostream &out = std::cout;
  RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));

// const global_size_t numGlobalElements = 200000000; 
   long long int trows ; 


   std::string matrixFileName ;

  ifstream myfile ("input.txt");
   if (myfile.is_open())
  {
   myfile >>  trows ;
   myfile >> matrixFileName ; 
  }


//  std::string matrixFileName = "wathenSmall.mtx";
  Teuchos::ParameterList defaultParameters;
  Teuchos::RCP <node_type> node = Teuchos::rcp(new node_type(defaultParameters));

  //Get Matrix
/*----------------Load a test matrix---------------*/
//  std::string matrixFileName = "wathenSmall.mtx";
 

  Teuchos::RCP<sparse_mat_type> A = Tpetra::MatrixMarket::Reader<sparse_mat_type>::readSparseFile(matrixFileName, comm, node); //removed node

//      if( &A == NULL)      success = false;


/*
//  myfile >> numGlobalElements >> '\n';
  const global_size_t numGlobalElements = trows ; 


  
  const size_t nnc = numGlobalElements/numProcs ; //map->getNodeNumElements ();
  size_t numMyElements = numGlobalElements/numProcs ; //map->getNodeNumElements ();
  if(myRank == numProcs-1) numMyElements = numGlobalElements -(numProcs-1)* nnc ; 

  ArrayRCP<size_t> glon = arcp<size_t> (numMyElements);

  double Values[7];
  int   Indices[7];
  int NumEntries;
  int RowLess3, RowLess2, RowLess1;
  int RowPlus1, RowPlus2, RowPlus3;

  int i;
  LO lcR;
  int gblRow;

  std::clock_t c_start1 = std::clock(); 

  RCP<const map_type> map = rcp (new map_type (numGlobalElements, numMyElements, 0 , comm));
  std::clock_t c_start2 = std::clock();  

  for(i = 0; i < numMyElements; ++i) {
      glon[i] = myRank + i *numProcs;
  }

//  RCP<crs_matrix_type> A (new crs_matrix_type (map, 7));
  RCP<sparse_mat_type> A (new sparse_mat_type (map, 7));

  ArrayRCP<size_t> aloc = arcp<size_t> (6);
  aloc[0] = -3 ; 
  aloc[1] = -2 ; 
  aloc[2] = -1 ; 
  aloc[3] =  1 ; 
  aloc[4] =  2 ; 
  aloc[5] =  3 ; 


  {
  TimeMonitor monitor (*insertva);
  for(lcR = 0; lcR < static_cast<LO> (numMyElements);  ++lcR) {
//    gblRow = glon[lcR]; //map->getGlobalElement (lcR);
    gblRow = map->getGlobalElement (lcR);
    NumEntries = 0; 
    RowLess3 = gblRow + aloc[0];
    RowLess2 = gblRow + aloc[1];
    RowLess1 = gblRow + aloc[2];
    RowPlus1 = gblRow + aloc[3];
    RowPlus2 = gblRow + aloc[4];
    RowPlus3 = gblRow + aloc[5];

    if ((RowLess3 >=0 )){
      Values[NumEntries]  = -0.1 ;  //Ao[lcR][0]; 
      Indices[NumEntries] = RowLess3; 
      NumEntries          = NumEntries + 1 ; 
    }

    if ((RowLess2 >=0 )) { 
      Values[NumEntries]  = -0.3; //Ao[lcR][1];
      Indices[NumEntries] = RowLess2;
      NumEntries          = NumEntries + 1 ;
    }
  
    if ((RowLess1 >=0 ))  {
      Values[NumEntries]  = -0.5;  //Ao[lcR][2];
      Indices[NumEntries] = RowLess1;
      NumEntries          = NumEntries + 1 ;
    }
  
    if ((RowPlus1 < numGlobalElements) )   {
       Values[NumEntries]  = -0.5; //Ao[lcR][4];
       Indices[NumEntries] = RowPlus1;
       NumEntries          = NumEntries + 1 ;
    }
    
   if ((RowPlus2 < numGlobalElements) )       {
      Values[NumEntries]  = -0.3;  //Ao[lcR][5];
      Indices[NumEntries] = RowPlus2;
      NumEntries          = NumEntries + 1 ;
   }
   
   if ((RowPlus3 < numGlobalElements))   {
      Values[NumEntries]  = -0.18;  //Ao[lcR][6];
      Indices[NumEntries] = RowPlus3;
      NumEntries          = NumEntries + 1 ;
      }

      Values[NumEntries]  = 1.8; //Ao[lcR][3];
      Indices[NumEntries] = gblRow;  
      NumEntries          = NumEntries + 1 ;

//       A->insertGlobalValues(gblRow,
//                       Teuchos::tuple<GO>(RowLess3, RowLess2, RowLess1, RowPlus1, RowPlus2, RowPlus3),
//                       Teuchos::tuple<ST>(-0.1,-0.3,-0.5,-0.5,-0.3,-0.15,2.0));
     
    A->insertGlobalValues (gblRow, NumEntries, Values, Indices); 
    }
  }

  {
    TimeMonitor monitor (*FillTimer);
    A->fillComplete (map,map);
  }

*/


  std::clock_t c_start3 = std::clock();

/*
  std::string inputFile="input_param.xml";
  std::string vname;
  std::string prectype;
  bool prec;
  RCP<Teuchos::ParameterList> myParams = rcp(new Teuchos::ParameterList());
  Teuchos::updateParametersFromXmlFile(inputFile, myParams.ptr());
  vname = myParams->get<std::string>("Solver");
  prectype = myParams->get<std::string>("PrecType");
  prec  = myParams->get<bool>("Precond");
*/

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
//


/*
   RCP<multivector_type> x = rcp(new multivector_type(map,1));
   RCP<multivector_type> b = rcp(new multivector_type(map,1));

// b.putScalar (STS::one ());
   b->randomize();
*/

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


/*
   for (LO lclRow = 0; lclRow < static_cast<LO> (numMyElements);++lclRow) {
    const GO gblRow = map->getGlobalElement (lclRow);
    b->sumIntoGlobalValue(gblRow, 0, bo[lclRow]);
//   x->sumIntoGlobalValue(gblRow, 0, x_vec[lclRow]);
   }

*/
   std::clock_t c_start4 = std::clock();

  std::string xmlFileName = "test.xml";
  
   std::clock_t c_start5 = std::clock();
//   TimeMonitor monitor(*makeprb); 
   RCP<linear_problem_type> Problem = rcp(new linear_problem_type(A, x, b));
   Problem->setProblem();

  std::clock_t c_start6 = std::clock();

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
/*
  std::string inputFile="input_param.xml";
  std::string vname, ortho; 
  bool prec;
  RCP<Teuchos::ParameterList> myParams = rcp(new Teuchos::ParameterList());
  Teuchos::updateParametersFromXmlFile(inputFile, myParams.ptr());
*/
  
//  std::clock_t c_start4 = std::clock();
//    RCP< mtoperator > mueLuPreconditioner = MueLu::CreateTpetraPreconditioner<ST,LO,GO,node_type>(RCP<operator_type>(A), xmlFileName);
//    Problem->setRightPrec (mueLuPreconditioner);
    Problem->setRightPrec (ifpack2Preconditioner);

//    solver = rcp(new belos_stdgmres_manager_type(Problem, belosList));
//      solver = rcp(new belos_flexgmres_manager_type(Problem, belosList));
    solver = rcp(new belos_blockcg_manager_type(Problem, belosList));

  solver->solve();
  std::clock_t c_start7 = std::clock();

}
