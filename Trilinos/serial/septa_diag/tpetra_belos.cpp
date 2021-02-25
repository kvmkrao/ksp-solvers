#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Tpetra_Version.hpp>

#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <iostream>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

#include <MueLu.hpp>

#include <MueLu_Exceptions.hpp>
#include <MueLu_Hierarchy.hpp>
#include <MueLu_MasterList.hpp>
#include <MueLu_MLParameterListInterpreter.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_Utilities.hpp>

#include "MueLu_FactoryManagerBase.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_PFactory.hpp"
#include "MueLu_SaPFactory.hpp"
#include "MueLu_TransPFactory.hpp"
#include "MueLu_RAPFactory.hpp"
#include "MueLu_AmesosSmoother.hpp"
#include "MueLu_TrilinosSmoother.hpp"
#include "MueLu_SmootherFactory.hpp"
#include "MueLu_CoupledAggregationFactory.hpp"
#include "MueLu_TentativePFactory.hpp"
#include "MueLu_AmesosSmoother.hpp"

#include <MueLu_Level.hpp>
#include <MueLu_MLParameterListInterpreter.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
//#ifdef HAVE_MUELU_TPETRA
#include <Tpetra_Operator.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <Xpetra_TpetraVector.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_UseDefaultTypes.hpp>

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
  typedef MueLu::TpetraOperator<scalar_type,local_ordinal_type,global_ordinal_type,node_type> mtoperator;
  
  typedef Tpetra::Operator<scalar_type,local_ordinal_type,global_ordinal_type,node_type>    operator_type;
  typedef Belos::LinearProblem<scalar_type, multivector_type, operator_type> linear_problem_type;
  typedef Belos::SolverManager<scalar_type, multivector_type, operator_type> belos_solver_manager_type;
  typedef Belos::BlockGmresSolMgr<scalar_type, multivector_type, operator_type> belos_gmres_manager_type;
  typedef Belos::BiCGStabSolMgr<scalar_type, multivector_type, operator_type> belos_bicgstab_manager_type;
  typedef Belos::BlockCGSolMgr<scalar_type, multivector_type, operator_type>    belos_blockcg_manager_type;
  typedef Belos::PseudoBlockCGSolMgr<scalar_type, multivector_type, operator_type> belos_pseudocg_manager_type;
  typedef Belos::TFQMRSolMgr<scalar_type, multivector_type, operator_type>      belos_tfqmr_manager_type;
  typedef Belos::LSQRSolMgr<scalar_type, multivector_type, operator_type>       belos_lsqr_manager_type;
  typedef MueLu::TpetraOperator<scalar_type, local_ordinal_type, global_ordinal_type, node_type> muelu_tpetra_operator_type;
  
  using Tpetra::global_size_t;
  using Teuchos::Array;
  using Teuchos::ArrayView;
  using Teuchos::ArrayRCP;
  using Teuchos::arcp;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  using std::cerr;
  using std::cout;
  using std::endl;
  using Teuchos::parameterList;
  using Teuchos::Time;
  using Teuchos::TimeMonitor;

  using Teuchos::updateParametersFromXmlFile;
  using Teuchos::updateParametersFromXmlString; 
  typedef Tpetra::CrsMatrix<> crs_matrix_type;


  Teuchos::oblackholestream blackhole;

  Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

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

  const global_size_t numGlobalElements = 10000;  // 50;

  const int nrows = numGlobalElements/numProcs; 
  size_t numMyElements = 10000 ; //map->getNodeNumElements ();

  /*
  if(myRank == numProcs-1) {
  numMyElements = numGlobalElements -myRank * nrows; 
  std::cout << " myRank"  << numMyElements << std::endl; 
  }
  */

//  int numMyElements =  rows; //map->getNodeNumElements ();
  ArrayRCP<size_t> glon = arcp<size_t> (numMyElements);

  const global_ordinal_type indexBase = 0;

//  double Values[7];
//  int    Indices[7];
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


/*
  RCP<const map_type> map;
 {
    Array<GO>::size_type numEltsPerProc = rows;
    Array<GO> myGlobalElements (numEltsPerProc); 
      for(int i = 0; i < numMyElements; ++i) {
      myGlobalElements[i] = glob[i] -1;  //myRank + k*numProcs;
    }
  map = rcp (new map_type (numGlobalElements, myGlobalElements, 0, comm));
}
*/
// map->describe(*fos,Teuchos::VERB_EXTREME);

  for(i = 0; i < numMyElements; ++i) {
      glon[i] = i;
  //    glon[i] = i + myRank*numMyElements;
  }
  double Ao[numMyElements][7];
  for (int i=0; i < numMyElements; ++i ) {
            Ao[i][0] =  -0.05;
            Ao[i][1] =  -0.1;
            Ao[i][2] =  -0.2;
            Ao[i][3] =   4.0;
            Ao[i][4] =  -0.2;
            Ao[i][5] =  -0.1;
            Ao[i][6] =  -0.05;
  }

  double aloc[numMyElements][6];
  for (int i=0; i < numMyElements; ++i ) {
            aloc[i][0] =  -100;
            aloc[i][1] =  -20;
            aloc[i][2] =  -1;
            aloc[i][3] =   1;
            aloc[i][4] =   20;
            aloc[i][5] =   100;
  }

  RCP<crs_matrix_type> A (new crs_matrix_type (map, 7));
//  std::clock_t c_start1 = std::clock();


  {
  TimeMonitor monitor (*insertva);
  for(lcR = 0; lcR < static_cast<LO> (numMyElements);  ++lcR) {
    gblRow = lcR; //glon[lcR]; //map->getGlobalElement (lcR);
    NumEntries = 0; 
    RowLess3 = gblRow + aloc[lcR][0];
    RowLess2 = gblRow + aloc[lcR][1];
    RowLess1 = gblRow + aloc[lcR][2];
    RowPlus1 = gblRow + aloc[lcR][3];
    RowPlus2 = gblRow + aloc[lcR][4];
    RowPlus3 = gblRow + aloc[lcR][5];


//    if ((RowLess3 >=0 ) && abs(Ao[lcR][0])!=0){
     if ((RowLess3 >=0 )&&  abs(aloc[lcR][0]) != 0.0 ) {
      Values[NumEntries]  = Ao[lcR][0];
      Indices[NumEntries] = RowLess3;
      NumEntries          = NumEntries + 1 ;
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowLess3),
             tuple<scalar_type> (Ao[lcR][0]));
      }

//      if (RowLess2 >=0 &&  RowLess2 !=GlobalRow) //aloc[i][1] != 0.0)
//    if ((RowLess2 >=0 ) && abs(Ao[lcR][1])!=0) { 
      if ((RowLess2 >=0 ) && abs(aloc[lcR][1]) != 0) {
      Values[NumEntries]  = Ao[lcR][1];
      Indices[NumEntries] = RowLess2;
      NumEntries          = NumEntries + 1 ;
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowLess2),
             tuple<scalar_type> (Ao[lcR][1]));

    }

//    if ((RowLess1 >=0 ) && abs(Ao[lcR][2])!=0)  {
      if ((RowLess1 >=0 ) && abs(aloc[lcR][2]) != 0) {
      Values[NumEntries]  = Ao[lcR][2];
      Indices[NumEntries] = RowLess1;
      NumEntries          = NumEntries + 1 ;
       A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowLess1),
             tuple<scalar_type> (Ao[lcR][2]));

    }

//       if (RowPlus1 < NumGlobalElements &&  RowPlus1 !=GlobalRow) //aloc[i][3] != 0.0)
//    if ((RowPlus1 < numGlobalElements && abs(Ao[lcR][4])!=0))   {
       if ((RowPlus1 < numGlobalElements) &&  abs(aloc[lcR][3]) != 0) {
      Values[NumEntries]  = Ao[lcR][4];
      Indices[NumEntries] = RowPlus1;
      NumEntries          = NumEntries + 1 ;
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowPlus1),
             tuple<scalar_type> (Ao[lcR][4]));

    }

      if ((RowPlus2 < numGlobalElements) &&  abs(aloc[lcR][4]) != 0) {
//   if ((RowPlus2 < numGlobalElements) && abs(Ao[lcR][5])!=0)       {
      Values[NumEntries]  = Ao[lcR][5];
      Indices[NumEntries] = RowPlus2;
      NumEntries          = NumEntries + 1 ;
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowPlus2),
             tuple<scalar_type> (Ao[lcR][5]));

   }
    if ((RowPlus3 < numGlobalElements) && abs(aloc[lcR][5]) != 0) {
//   if ((RowPlus3 < numGlobalElements) && abs(Ao[lcR][6])!=0)   {
      Values[NumEntries]  = Ao[lcR][6];
      Indices[NumEntries] = RowPlus3;
      NumEntries          = NumEntries + 1 ;
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (RowPlus3),
             tuple<scalar_type> (Ao[lcR][6]));
      }

      Values[NumEntries]  = Ao[lcR][3];
      Indices[NumEntries] = gblRow;
      NumEntries          = NumEntries + 1 ;
//      A->insertGlobalValues (gblRow, NumEntries, Values, Indices);
      A->insertGlobalValues (gblRow,
             tuple<global_ordinal_type> (gblRow),
             tuple<scalar_type> (Ao[lcR][3]));

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

//  RCP<multivector_type> x(map);
//  RCP<multivector_type> b(map);

//  b.putScalar (STS::one ());
    b->randomize();

/*
   for (LO lclRow = 0; lclRow < static_cast<LO> (numMyElements);++lclRow) {
    const GO gblRow = map->getGlobalElement (lclRow);
    b->sumIntoGlobalValue(gblRow, 0, bo[lclRow]);
//   x->sumIntoGlobalValue(gblRow, 0, x_vec[lclRow]);
   }
*/

   std::clock_t c_start4 = std::clock();

/*
  Teuchos::ParameterList paramList;
  paramList.set("verbosity", "low");
  paramList.set("max levels", 2);
  paramList.set("coarse: max size", 10);
  paramList.set("multigrid algorithm", "sa");
*/
  
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
  
  std::string inputFile="input_param.xml";
  std::string vname; 
  bool prec;
  RCP<Teuchos::ParameterList> myParams = rcp(new Teuchos::ParameterList());
  Teuchos::updateParametersFromXmlFile(inputFile, myParams.ptr());
  vname = myParams->get<std::string>("Solver");
  prec  = myParams->get<bool>("Precond");
  
//  std::clock_t c_start4 = std::clock();
  if(vname=="GMRES") {
    if(prec) {
    RCP< mtoperator > mueLuPreconditioner = MueLu::CreateTpetraPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type>(RCP<operator_type>(A), xmlFileName);
    Problem->setLeftPrec (mueLuPreconditioner);
    }
    solver = rcp(new belos_gmres_manager_type(Problem, belosList));
    }
    else if(vname=="CG")  {
    if(prec) {
    RCP< mtoperator > mueLuPreconditioner = MueLu::CreateTpetraPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type>(RCP<operator_type>(A), xmlFileName);
    Problem->setRightPrec (mueLuPreconditioner);
    }
    solver = rcp(new belos_blockcg_manager_type(Problem, belosList));
    }
    else if(vname=="TFQMR")     {
     solver = rcp(new belos_tfqmr_manager_type(Problem, belosList));
    }
//    else if(vname=="BICG") { 
//    solver = rcp(new belos_bicgiter_manager_type(Problem, belosList)); 
//    }
    else if(vname=="LSQR")     {
     solver = rcp(new  belos_lsqr_manager_type(Problem, belosList));
    }

    else
    {
    if(prec) {
    RCP< mtoperator > mueLuPreconditioner = MueLu::CreateTpetraPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type>(RCP<operator_type>(A), xmlFileName);
//    Problem->setLeftPrec (mueLuPreconditioner);
    Problem->setRightPrec (mueLuPreconditioner);
    }
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


/*  ArrayRCP<ST> view;
  int size = x->getLocalLength ();
  Array<ST> copy1(numMyElements);
  view = x->get1dViewNonConst();
  x->get1dCopy(copy1(),numMyElements);
  
  for(int i=0; i < numMyElements; i++) {
    x_vec[i] = copy1[i]; 
//      std::cout << myRank  << "  " << glob[i] << " " << i <<"  " <<copy1[i] << std::endl;
  }
*/


  TimeMonitor::summarize();
  return 0; 
}  // end of cpp wrapper 

// *fos << "LHS :" << std::endl;
//  x->describe(*fos,Teuchos::VERB_EXTREME);

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_config.h"

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>

extern "C" {

  void xmlstrc_(char *varname, int *varlen, char *Str1, int *len)  {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::parameterList;
  Teuchos::oblackholestream blackHole;
  Teuchos::GlobalMPISession mpiSession();

  std::string vname;
  vname.assign(varname, *varlen);

  std::string name1;
  name1.assign(Str1, *len);

  std::string inputFile;
  inputFile="input_param.xml";
  RCP<Teuchos::ParameterList> myParams = rcp(new Teuchos::ParameterList());
  Teuchos::updateParametersFromXmlFile(inputFile, myParams.ptr());
  vname = myParams->get<std::string>(name1);
//  std::cout << vname << name1 << std::endl;
  strncpy(varname, vname.c_str(), *varlen);
  varname[*varlen-1] = 0 ;
//  std::cout << vname.c_str() << std::endl;
 }
}


#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_config.h"

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>

extern "C" {
  void xmlintc_(int *intvar1, char *Str1, int *len)  {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::parameterList;

  Teuchos::oblackholestream blackHole;
  Teuchos::GlobalMPISession mpiSession();

  std::string name1;
  name1.assign(Str1, *len);
  std::string inputFile;
  inputFile="input_param.xml";
  RCP<Teuchos::ParameterList> myParams = rcp(new Teuchos::ParameterList());
  Teuchos::updateParametersFromXmlFile(inputFile, myParams.ptr());
   *intvar1 = myParams->get<int>(name1);
  }
}


#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_config.h"

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>

extern "C" {
  void xmlfltc_(double *intvar1, char *Str1, int *len)  {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::parameterList;

  Teuchos::oblackholestream blackHole;
  Teuchos::GlobalMPISession mpiSession();

  std::string name1;
  name1.assign(Str1, *len);
  std::string inputFile;
  inputFile="input_param.xml";
  RCP<Teuchos::ParameterList> myParams = rcp(new Teuchos::ParameterList());
  Teuchos::updateParametersFromXmlFile(inputFile, myParams.ptr());
   *intvar1 = myParams->get<double>(name1);
//   std::cout << *intvar1 << name1 << std::endl;
  }
}
