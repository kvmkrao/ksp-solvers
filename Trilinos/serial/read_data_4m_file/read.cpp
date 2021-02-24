#include <fstream>
#include <iostream>
#include <stdlib.h> 
using namespace std;

double itersolve(int &ent, int &nn, int *row, int *col, double *val, double *rhs) ;

int main() {
    string word; 
    int rmax,cmax,i;  
    int ii, nodes, nmax = 50;  
//    int row[nmax], col[nmax];
//    float  value[nmax],rhs[nmax]; 

    ifstream infile; 
    infile.open("val_S0.dat"); 
//    if (!file.is_open()) return -1;

     if(!infile) {
	     cout << "Error in opening the file" << std::endl; 
     }

     system("wc -l val_S0.dat");

     cout << "enter number of rows" <<std::endl; 
     cin >> nmax;  

     infile >> word >> word >> word ;

     int row[nmax], col[nmax];
     double  value[nmax],rhs[nmax];

//     while (!infile.eof( ))    //if not at end of file, continue reading numbers
     for(int i=0; i<nmax; i++) {
          infile >> row[i] >> col[i] >> value[i];
	   cout << i << " " << row[i] << " "<< col[i] << std::endl; 
	   if(i >nmax) break;  
     }

    infile.close();
    rmax = 0; 
    cmax = 0; 
    for(int i=0; i< nmax; i++) {
	    if(row[i] >rmax) rmax = row[i]; 
    }

    cmax = 0; 
    for(int i=0; i< nmax; i++) {
	    if(col[i]> cmax) cmax = col[i];
    } 

    cout << rmax <<" "<<cmax <<std::endl;  

    ifstream rfile;
    rfile.open("rhs_S0.dat");

     if(!rfile) {
             cout << "Error in opening the file" << std::endl;
     }

     rfile >> word >> word ;
     for(int i=0; i< nodes; i++) {
          rfile >> ii >> rhs[ii] ;
//          cout << ii << " " << rhs[i] << std::endl;
     }
     rfile.close();

    itersolve(nmax,rmax,row,col,value, rhs);
    // Prints sum 

    return 0;
}
