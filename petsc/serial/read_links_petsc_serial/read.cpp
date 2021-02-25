#include <fstream>
#include <iostream>
using namespace std;

int itersolve(int &nn, double *rad,double  *len, int (*link)[2]) ;

int main() {
    int nn; 
    int i1, i2, i3, i4,ii;
    double a,b,c,d,rd, vol, sfact, nd1len,nd2len, xyz,lklen, radius,dis, linkn1,linkn2;  

    ifstream infile; 
    infile.open("link.dat"); 
//    if (!file.is_open()) return -1;

     if(!infile) {
	     cout << "Error in opening the file" << std::endl; 
     }

    infile >> nn;
    int link[nn][2];
    double rad[nn],length[nn];

//link id, node 1 id (0 means inlet or outlet), node 2 id, position (0 = inside, 1 = inlet, 2 = outlet) radius, volume, shape factor,  node1 length, node 2 length, link lenght, distance (node 1 center to node 2 center, useless info so far)

     ii=0; 
     for(int i=0; i< nn; i++) {
          infile >> i1 >> linkn1 >> linkn2 >> i2 >> radius >> vol >> sfact >> nd1len >> nd2len >> lklen >> dis >> xyz ;
	  if(i2 == 0) { 
            link[ii][0] = linkn1;
            link[ii][1] = linkn2;
            rad[ii]     = radius; 
	    length[ii]  = dis; 
	    ii = ii+1;
	  }
//	   cout << i << " " << link[i1][0] << " "<< link[i1][1] <<" "<< rad[i1] << " "<< length[i1] << std::endl; 
//          cout << row[i] << col[i] << value[i] << endl; 
     }
     infile.close();

    cout << nn << endl;
  //  return 0; 
    itersolve(ii,rad,length,link);
    // Prints sum 

    return 0;
}
