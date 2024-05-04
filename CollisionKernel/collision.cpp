int NNmax = 20;

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <iomanip>
#include <chrono>
using namespace std;

//#include "./QmatrixPA4_unrotated.cpp"
//#include "./QmatrixPA6_unrotated.cpp"
#include "./QmatrixPA8_unrotated.cpp"


struct S{
    double coefficient;
    int caseNumber;
    int startIndexResult;
    int startIndexLeft;
    int startIndexRight;
        
    S(double coeff, int caseN, int startIRes, int startIL, int startIR): coefficient(coeff), caseNumber(caseN), startIndexResult(startIRes), startIndexLeft(startIL), startIndexRight(startIR)
    {
//        cout << "S is initialized for " << coefficient << "\t" << "\t" << caseN << "\t" << "\t" << startIndexResult << "\t" << startIndexLeft << "\t" << startIndexRight << "\n";
    }
};

void readSinit(ifstream &Sinit, int* &aMax,  int &nMax){
    int init[2]={0,0};
    string line;
    getline(Sinit, line);
    stringstream nvalue(line);
    nvalue >> nMax;
    
    aMax = new int[nMax];
    getline(Sinit,line);
    stringstream aValues(line);
    for(int i=0; i<nMax; i++){
        aValues >> aMax[i];        
    }
    nMax--; 
}


int main(int argc, char** argv) 
{
    bool verbalMode=false; //if false, only timing will be reported
    string locationSinit;
    string locationSvalues;
    string locationFinit;
    string locationSout;
    string locationFout;
	double deltaT;
	int n;
    int *aMax, nMax, Slength = 0;
    int reportAllValues;
    vector<S> Smatrix;
    if(argc < 2){
        cout << "Please state a preferences file\n";
        return 0;
    }
    if(argc>2){
        if(!strcmp(argv[2], "1") || !strcmp(argv[2], "true")){
            verbalMode = true;
            cout << "Verbal Mode on\n";
        }            
    }
    // get the file locations for the Smatrix
    ifstream preferences (argv[1]);
    if (preferences.is_open()){
        getline(preferences, locationSinit);
        getline(preferences, locationSvalues);
        getline(preferences, locationFinit);
        getline(preferences, locationSout);
        getline(preferences, locationFout);
		preferences >> deltaT;
		preferences >> n;
		preferences >> reportAllValues;
//        getline(preferences, deltaT);
//        getline(preferences, n);
        preferences.close();
        
    }else{
        cout << "Couldn't open preferences file.\n Aborting Calculation.\n" ;
        return 0;
    }        
    
    // read out the initialization parameters aMax, nMax, Slength from the Sinit-file
    ifstream Sinit(locationSinit);
    if(Sinit.is_open()){
        readSinit(Sinit, aMax, nMax);
        Sinit.close();
    }else{
        cout << "Couldn't open S-Matrix initialization file, check your preferences.\nAborting Calculation.\n";
        return 0;
    }
    
    
    // initialize the vector for the SMatrix: Scoefficients (which contains all the physical information) and Scases (which maps the coefficients to the correct Q and start indexes of f and fprime)
    
    ifstream Smatrixfile (locationSvalues);
    if (Smatrixfile.is_open()){
        string line;
        double coeff;
        int para[4] = {0, 0, 0, 0};
        while(getline(Smatrixfile, line)) {
            stringstream smvalues(line);
            smvalues >> coeff >> para[0] >> para[1] >> para[2] >> para[3];        
            Smatrix.push_back(S(coeff, para[0], para[1], para[2], para[3]));  
            Slength++;
        }
        Smatrixfile.close();
        //the last element of Smatrix is added twice, removing it now and setting Slength to the correct value
        Slength--;
        Smatrix.pop_back();
    }else{
        cout << "Couldn't open S-Matrix file, check your preferences.\n Aborting Calculation\n";
        return 0;
    }    

    
    ifstream Finitfile (locationFinit);
    if (Finitfile.is_open()){
        // allocate and initialize the distribution function of current and next timestep and the collision operator as a one-dimensional array with a particular order of the parameters - similar to [n][a][l]
        // Calculate the size of the array based on the input from Sinit:
        int totalSize = 0;
        for (int l = 0; l <= nMax; l++) {
            totalSize += (1+aMax[l])*(2*l+1);
        }
        //totalSize *= (aMax+1);
        
		double* f0 = new double[totalSize];
        double* Soperator = new double[totalSize];
		double h = deltaT/n;
		double* tempPointer;
        
        double** foft;
        // intitialize f to 0
        if (verbalMode) cout << "Initializing f at t=0 ... \n";
        for (int j = 0;  j < totalSize;  j++) {
			f0[j] = 0;
        }
        
        //if you want to have all values of t reported, fort needs to be initialized. It is of dimensions (n+1) times (totalSize+1). The zeroth entry in each n is for the time.
        if(reportAllValues){
            foft = new double*[n+1];
        for (int j = 0;  j <= n;  j++) {
            foft[j] = new double[totalSize+1];
        }
        for(int j=0; j<=n; j++){
            for(int k=0; k<=totalSize; k++){
                foft[j][k]=0;
            }
        }
            
        }
        // initialize the values of fnow that aren't 0 as given in the Finitfile
        string line;
        double coeff;
        int idx;
        if (verbalMode) cout << "Reading values for f at t=0 from file...\n";
        while(getline(Finitfile, line)) {
            stringstream finitvalues(line);
            finitvalues >> idx >> coeff;        
			f0[idx] = coeff;
            if(reportAllValues) foft[0][idx+1]=coeff;
        }
        Finitfile.close();
		
		
		// For testing purposes: Get the value of the S-Vector at t=0:
		// Step 1: initialize Soperator to 0 everywhere
		for (int j=0; j < totalSize; j++){
            Soperator[j] = 0;			
		}
		// Step 2: calculate the S-Vector with the initial distribution function.
        for (int j = 0;  j < Slength;  j++) {
            calcQ(Smatrix[j].caseNumber, &f0[Smatrix[j].startIndexLeft], &f0[Smatrix[j].startIndexRight], &Soperator[Smatrix[j].startIndexResult], Smatrix[j].coefficient);
        }

        for(int i = 0; i < totalSize; i++) {
            std::cout << i << " " << Soperator[i] << std::endl;
        }
	}

    
}
