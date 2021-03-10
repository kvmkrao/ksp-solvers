/* Author: VMK Kotteda 
 * date  : Feb 18, 2021 */

#include <fstream>
#include <iostream>
using namespace std;

//int itersolve(int &nn, int &ent, int *row, int *col,double *rhs, double *xi, double *val) ;

int main() {
    string word;   
    int ii, nodes = 288420, max = 1368756;  
    int row[max], col[max];
    double value[max],rhs[nodes]; //,xi[nodes]; 

    ifstream matfile; 
    matfile.open("sparsemat_S0.dat"); 

     if(!matfile) {
	     cout << "Error in opening the file" << std::endl; 
     }

     matfile >> word >> word >> word ;
     for(int i=0; i< max; i++) {
          matfile >> row[i] >> col[i] >> value[i];
//	   cout << i << " " << row[i] << " "<< col[i] << std::endl; 
//          cout << row[i] << col[i] << value[i] << endl; 
     }
     matfile.close();

/*     
    ifstream rfile,xfile;
    rfile.open("rhs_S0.dat");
    xfile.open("initp_S0.dat");

    // read right hand side vector 
     if(!rfile) {
             cout << "Error in opening the file" << std::endl;
     }

     rfile >> word >> word ;
     for(int i=0; i< nodes; i++) {
          rfile >> ii >> rhs[i] ;
          cout << ii << " " << rhs[i] << std::endl;
     }
     rfile.close();
*/
     /*
     // read initial values      
     if(!xfile) {
             cout << "Error in opening the file" << std::endl;
     }

     xfile >> word ;
     for(int i=0; i< nodes; i++) {
          xfile >> xi[i] ;
          cout << i << " " << rhs[i] << std::endl;
     }
     xfile.close();
     */
//     itersolve(nodes,max,row,col,rhs,xi,value);

    return 0;
}
