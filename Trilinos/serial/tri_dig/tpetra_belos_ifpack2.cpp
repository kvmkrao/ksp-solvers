#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <iostream>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Core.hpp>

#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

#include <ctime>
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
using namespace Teuchos;
using namespace std; 

int main(int argc, char* argv[])
{

  typedef Tpetra::MultiVector<>::scalar_type scalar_type;
  typedef scalar_type ST;
  typedef Tpetra::MultiVector<>::local_ordinal_type LO;
  typedef Tpetra::MultiVector<>::global_ordinal_type GO;
  //typedef Tpetra::MultiVector<>::node_type node_type;
  
  typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
  
  typedef Teuchos::ScalarTraits<scalar_type> STS;
//  typedef STS::magnitudeType magnitude_type;
//  typedef Teuchos::ScalarTraits<magnitude_type> STM;
  
  typedef Tpetra::Map<LO, GO, node_type> map_type;
  typedef Tpetra::MultiVector<scalar_type, LO, GO, node_type> multivector_type;
  typedef Tpetra::CrsMatrix<scalar_type, LO, GO, node_type> sparse_mat_type;
  
  typedef Tpetra::Vector<>::scalar_type scalar_type;
  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  
  typedef Tpetra::Operator<scalar_type,local_ordinal_type,global_ordinal_type,node_type>    operator_type;
  typedef Belos::LinearProblem<scalar_type, multivector_type, operator_type> linear_problem_type;
  typedef Belos::SolverManager<scalar_type, multivector_type, operator_type> belos_solver_manager_type;
  typedef Belos::BlockGmresSolMgr<scalar_type, multivector_type, operator_type> belos_gmres_manager_type;
  typedef Belos::BiCGStabSolMgr<scalar_type, multivector_type, operator_type> belos_bicgstab_manager_type;
  typedef Belos::BlockCGSolMgr<scalar_type, multivector_type, operator_type>    belos_blockcg_manager_type;
  typedef Belos::PseudoBlockCGSolMgr<scalar_type, multivector_type, operator_type> belos_pseudocg_manager_type;
  typedef Belos::TFQMRSolMgr<scalar_type, multivector_type, operator_type>      belos_tfqmr_manager_type;
  typedef Belos::LSQRSolMgr<scalar_type, multivector_type, operator_type>       belos_lsqr_manager_type;
  typedef Ifpack2::Preconditioner<ST,LO,GO,node_type> prec_type; 
  std::cout << "Execution space name: ";

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
  using Teuchos::Comm;


//  Teuchos::oblackholestream blackhole;
//  Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
//  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
//  RCP<const Teuchos::Comm<int> > comm = Teuchos::RCP<const Teuchos::Comm<int> >getDefaultComm ();

  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm ();

//  Teuchos::oblackholestream blackHole;
//  Teuchos::GlobalMPISession mpiSession (NULL,NULL, &blackHole);

  std::clock_t c_start = std::clock();
//  Teuchos::GlobalMPISession mpiSession();
//  RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  const int myRank = comm->getRank ();
  const int numProcs = comm->getSize ();

  RCP<Time> insertva     = TimeMonitor::getNewCounter ("InsertValues ");
  RCP<Time> FillTimer    = TimeMonitor::getNewCounter ("FillComplete A");
//  RCP<Time> makevec      = TimeMonitor::getNewCounter ("Make Vecs ");
//  RCP<Time> makeprb      = TimeMonitor::getNewCounter ("Problem setup ");
//  RCP<Time> SolsTimer    = TimeMonitor::getNewCounter ("SolverSetupTime ");

  std::ostream &out = std::cout;
  RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));


  std::string prectype, inputFile="input_param.xml";
  int nn; 
  RCP<Teuchos::ParameterList> myParams = rcp(new Teuchos::ParameterList());
  Teuchos::updateParametersFromXmlFile(inputFile, myParams.ptr());
  nn = myParams->get<int>("rows");

  const global_size_t numGlobalElements = nn;  // 50;

  const int nrows = numGlobalElements; //numProcs; 
  size_t numMyElements = nn ; //map->getNodeNumElements ();

  /*
  if(myRank == numProcs-1) {
  numMyElements = numGlobalElements -myRank * nrows; 
  std::cout << " myRank"  << numMyElements << std::endl; 
  }
  */

//  int numMyElements =  rows; //map->getNodeNumElements ();
  ArrayRCP<size_t> glon = arcp<size_t> (numMyElements);

  const global_ordinal_type indexBase = 0;

  int NumEntries;
  int RowLess3, RowLess2, RowLess1;
  int RowPlus1, RowPlus2, RowPlus3;

  int i;
  LO lcR;
  int gblRow;

  std::clock_t c_start1 = std::clock(); 

  RCP<const map_type> map = rcp (new map_type (numGlobalElements, numMyElements, indexBase, comm));
  std::clock_t c_start2 = std::clock();  

  Array<global_ordinal_type> Indices(7);
  Array<scalar_type> Values(7);

//  std::cout << " rank" << myRank << " " << numMyElements << std::endl; 
//  return 0; 


  double Ao[numMyElements][3];
  for (int i=0; i < numMyElements; ++i ) {
            Ao[i][0] =  -1.0;
            Ao[i][1] =  -2.0;
            Ao[i][2] =  -1.0;
  }
  double aloc[numMyElements][6];
  for (int i=0; i < numMyElements; ++i ) {
            aloc[i][0] =  -1;
            aloc[i][1] =   1;
  }
  RCP<sparse_mat_type> A (new sparse_mat_type (map, 3));
//  std::clock_t c_start1 = std::clock();


  {
  TimeMonitor monitor (*insertva);
  for(lcR = 0; lcR < static_cast<LO> (numMyElements);  ++lcR) {
    gblRow = lcR; //glon[lcR]; //map->getGlobalElement (lcR);
    NumEntries = 0; 
    RowLess1 = gblRow + aloc[lcR][0];
    RowPlus1 = gblRow + aloc[lcR][1];

//    if ((RowLess1 >=0 ) && abs(Ao[lcR][2])!=0)  {
      if ((RowLess1 >=0 ) ) {
      Values[NumEntries]  = Ao[lcR][0];
      Indices[NumEntries] = RowLess1;
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowLess1),
             tuple<scalar_type> (Ao[lcR][0]));

    }

      if ((RowPlus1 < numGlobalElements)) {
      Values[NumEntries]  = Ao[lcR][2];
      Indices[NumEntries] = RowPlus1;
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowPlus1),
             tuple<scalar_type> (Ao[lcR][2]));

    }

      Values[NumEntries]  = Ao[lcR][1];
      Indices[NumEntries] = gblRow;
//      A->insertGlobalValues (gblRow, NumEntries, Values, Indices);
      A->insertGlobalValues (gblRow,
         tuple<global_ordinal_type> (gblRow),
         tuple<scalar_type> (Ao[lcR][1]));

//    std::cout << gblRow << " "  << NumEntries << " "<< Indices << std::endl; 
    }
  }

  {
    TimeMonitor monitor (*FillTimer);
//  A->fillComplete ();
    A->fillComplete (map,map);
  }


// A->describe(*fos,Teuchos::VERB_EXTREME);
  std::clock_t c_start3 = std::clock();


//   TimeMonitor monitor(*makevec);
   RCP<multivector_type> x = rcp(new multivector_type(map,1));
   RCP<multivector_type> b = rcp(new multivector_type(map,1));

   x->putScalar (0.5*STS::one ());
   b->putScalar (STS::one ());
//  b->randomize();


/*   
   for (LO lclRow = 0; lclRow < static_cast<LO> (numMyElements);++lclRow) {
    const GO gblRow = map->getGlobalElement (lclRow);
    b->sumIntoGlobalValue(gblRow, 0, bo[lclRow]);
//   x->sumIntoGlobalValue(gblRow, 0, x_vec[lclRow]);
   }
*/

   std::clock_t c_start4 = std::clock();
   
  prectype = myParams->get<string>("Precond_type");
  std::string ifpack2par="ifpackpre.xml";
  Ifpack2::Factory factory;
  //DIAGONAL", "RELAXATION", "CHEBYSHEV", "ILUT", "RILUK"
  //RCP<prec_type> ifpack2Preconditioner = factory.create<crs_matrix_type> ("CHEBYSHEV", A);
  RCP<prec_type> ifpack2Preconditioner = factory.create<sparse_mat_type> (prectype, A);
  //ifpack2Preconditioner->setParameters( paramList );
  ifpack2Preconditioner->setParameters( ifpack2par);
  ifpack2Preconditioner->initialize();
  ifpack2Preconditioner->compute();
  
  std::string xmlFileName = "test.xml";
  
   std::clock_t c_start5 = std::clock();
//   TimeMonitor monitor(*makeprb); 
   RCP<linear_problem_type> Problem = rcp(new linear_problem_type(A, x, b));
   Problem->setProblem();

  std::clock_t c_start6 = std::clock();

 //  TimeMonitor monitor(*SolsTimer) ;   
//  RCP<ParameterList> belosList = rcp(new ParameterList());
  RCP<Teuchos::ParameterList> belosList = rcp(new Teuchos::ParameterList());
  belosList->set("Maximum Iterations",    20);    // Maximum number of iterations allowed
  belosList->set( "Num Blocks", 200);                // Maximum number of blocks in Krylov factorization
  belosList->set("Convergence Tolerance", 1.0e-4);     // Relative convergence tolerance requested
  belosList->set("Block Size",          1);
  //  belosList->set("Verbosity",             Belos::Errors + Belos::Warnings + Belos::StatusTestDetails + Belos::TimingDetails);
  belosList->set("Verbosity",             Belos::Errors + Belos::Warnings);
  belosList->set("Output Frequency",      20);
  //  belosList->set("Output Style",          Belos::None);
  belosList->set("Output Style",          Belos::Brief);
  //  belosList->set("Implicit Residual Scaling", "None");
  RCP<belos_solver_manager_type> solver;
  Problem->setRightPrec (ifpack2Preconditioner);

  std::string vname; 
  bool prec;
  vname = myParams->get<std::string>("Solver");
  prec  = myParams->get<bool>("Precond");
  
//  std::clock_t c_start4 = std::clock();
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

  if(myRank == 0 ) {
  std::cout << "Initializing arrays "  << 1000*(c_start1-c_start)/CLOCKS_PER_SEC   << std::endl; 
  std::cout << "Creating Map  "        << 1000*(c_start2-c_start1)/CLOCKS_PER_SEC   << std::endl; 
  std::cout << "Make vecs  "           << 1000*(c_start4-c_start3)/CLOCKS_PER_SEC   << std::endl; 
  std::cout << "IntiMapFormMatVec "    << 1000*(c_start5-c_start1)/CLOCKS_PER_SEC   << std::endl;
  std::cout << "Problem Setup  "       << 1000*(c_start6-c_start5)/CLOCKS_PER_SEC   << std::endl; 
  std::cout << "Solver  "              << 1000*(c_start7-c_start6)/CLOCKS_PER_SEC   << std::endl; 
  
/*  std::cout << "before fill "   << 1000*(c_start1-c_start)/CLOCKS_PER_SEC   << std::endl; 
  std::cout << " Fill "         << 1000*(c_start2-c_start1)/CLOCKS_PER_SEC << std::endl; 
  std::cout << " Fill b "       << 1000*(c_start3-c_start2)/CLOCKS_PER_SEC << std::endl; 
  std::cout << " before Solve " << 1000*(c_start4-c_start3)/CLOCKS_PER_SEC << std::endl; 
  std::cout << "Solve "         << 1000*(c_start5-c_start4)/CLOCKS_PER_SEC << std::endl; 
  std::cout << "total"          << 1000*(c_start5-c_start)/CLOCKS_PER_SEC << std::endl;
*/
  int numIterations = solver->getNumIters();
  std::cout << " number of iterations" <<  numIterations << std::endl;
  }


  ArrayRCP<ST> view;
  int size = x->getLocalLength ();
  Array<ST> copy1(numMyElements);
  view = x->get1dViewNonConst();
  x->get1dCopy(copy1(),numMyElements);
  
  for(int i=0; i < numMyElements; i++) {
    x_vec[i] = copy1[i]; 
    std::cout << myRank  << "  " << glob[i] << " " << i <<"  " <<copy1[i] << std::endl;
  }



  TimeMonitor::summarize();
  return 0; 
}  // end of cpp wrapper 

