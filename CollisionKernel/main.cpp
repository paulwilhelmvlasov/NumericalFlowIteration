int NNmax = 20;

#include "header.h"
//using namespace std;
#include <chrono>

// NB: ALL OF THESE 3 FUNCTIONS BELOW USE SIGNED VALUES INTERNALLY AND WILL EVENTUALLY OVERFLOW (AFTER 200+ YEARS OR
// SO), AFTER WHICH POINT THEY WILL HAVE *SIGNED OVERFLOW*, WHICH IS UNDEFINED BEHAVIOR (IE: A BUG) FOR C/C++.
// But...that's ok...this "bug" is designed into the C++11 specification, so whatever. Your machine won't run for 200
// years anyway...

// Get time stamp in milliseconds.
uint64_t millis()
{
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::
                  now().time_since_epoch()).count();
    return ms; 
}

// Get time stamp in microseconds.
uint64_t micros()
{
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::
                  now().time_since_epoch()).count();
    return us; 
}

// Get time stamp in nanoseconds.
uint64_t nanos()
{
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::
                  now().time_since_epoch()).count();
    return ns; 
}

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
    //return init;   
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
        
        double* fnow = new double[totalSize];
		double* f0 = new double[totalSize];
        double* fnext = new double[totalSize];
        double* Soperator = new double[totalSize];
		double* k1 = new double[totalSize];
		double* k2 = new double[totalSize];
		double* k3 = new double[totalSize];
		double* k4 = new double[totalSize];
		double* kf1 = new double[totalSize];
		double* kf2 = new double[totalSize];
		double* kf3 = new double[totalSize];
		double h = deltaT/n;
		double* tempPointer;
        
        double** foft;
        // intitialize f to 0
        if (verbalMode) cout << "Initializing f at t=0 ... \n";
        for (int j = 0;  j < totalSize;  j++) {
            fnow[j] = 0;
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
            fnow [idx] = coeff;
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
		time_t start, end; 
        uint64_t start_ms, end_ms;
        uint64_t start_ns, end_ns;
        if (verbalMode) cout << "Calculating solution to homogeneous Boltzmann Equation...\n";
  
		/* You can call it like this : start = time(NULL); 
		in both the way start contain total time in seconds  
		since the Epoch. */
		time(&start); 
        start_ms = micros();
        start_ns = nanos();
		
		// For testing purposes: crank up the counter and measure the time taken to compare.
		// Result for 21100: 76s in Andrea's Basis, 84s in Manuels Basis with counter up to 100
		for(int counter = 0; counter < 1; counter++){
		// unsync the I/O of C and C++. 
		ios_base::sync_with_stdio(false); 
		
  
   		for(int i=0; i<n; i++){
        // intitialize everything else to 0
		for (int j=0; j < totalSize; j++){
            fnext[j] = 0;
			k1[j]=0;
			k2[j]=0;
			k3[j]=0;
			k4[j]=0;
			kf1[j]=0;
			kf2[j]=0;
			kf3[j]=0;
			
		}
		//Calculate the four factors for the Runge-Kutta 4 iteration step
		//calculate k1
        for (int j = 0;  j < Slength;  j++) {
            calcQ(Smatrix[j].caseNumber, &fnow[Smatrix[j].startIndexLeft], &fnow[Smatrix[j].startIndexRight], &k1[Smatrix[j].startIndexResult], Smatrix[j].coefficient);
        }
		//calculate kf1
 		for (int j=0; j < totalSize; j++){
			kf1[j]=fnow[j] + h*0.5*k1[j];			
		}
		//calculate k2
        for (int j = 0;  j < Slength;  j++) {
            calcQ(Smatrix[j].caseNumber, &kf1[Smatrix[j].startIndexLeft], &kf1[Smatrix[j].startIndexRight], &k2[Smatrix[j].startIndexResult], Smatrix[j].coefficient);
        }
		//calculate kf2
 		for (int j=0; j < totalSize; j++){
			kf2[j]=fnow[j] + h*0.5*k2[j];			
		}
		//calculate k3
        for (int j = 0;  j < Slength;  j++) {
            calcQ(Smatrix[j].caseNumber, &kf2[Smatrix[j].startIndexLeft], &kf2[Smatrix[j].startIndexRight], &k3[Smatrix[j].startIndexResult], Smatrix[j].coefficient);
        }
		//calculate kf3
 		for (int j=0; j < totalSize; j++){
			kf3[j]=fnow[j] + h*k3[j];			
		}
		//calculate k4
        for (int j = 0;  j < Slength;  j++) {
            calcQ(Smatrix[j].caseNumber, &kf3[Smatrix[j].startIndexLeft], &kf3[Smatrix[j].startIndexRight], &k4[Smatrix[j].startIndexResult], Smatrix[j].coefficient);
        }
		//calculate fnext
 		for (int j=0; j < totalSize; j++){
			fnext[j]=fnow[j] + h/6.*k1[j] + h/3.*k2[j] + h/3.*k3[j] + h/6.*k4[j];
            if(reportAllValues){
                foft[i+1][0] = h*(i+1);
                foft[i+1][j+1] = fnext[j];
            }
		}
       
        // Calculate the result of the collision operator,  save it in Soperator:
//        for (int j = 0;  j < Slength;  j++) {
//            calcQ(Smatrix[j].caseNumber, &fnow[Smatrix[j].startIndexLeft], &fnow[Smatrix[j].startIndexRight], &Soperator[Smatrix[j].startIndexResult], Smatrix[j].coefficient);
//        }
		
		tempPointer = fnow;
		fnow = fnext;
		fnext = tempPointer;
		}
	}
         // Recording end time. 
		time(&end); 
        end_ms = micros();
        end_ns = nanos();
  

		// Write the result into the output files Sout and Fout and print them to the console
		ofstream SoutFile (locationSout);
		ofstream FoutFile (locationFout);
		
		if(SoutFile.is_open()){
			if(FoutFile.is_open()){
                if (verbalMode) cout << n << "\t\t\t" << deltaT << "\t\t\t" << reportAllValues <<  "\n\n\n";
                if(reportAllValues){
                    for(int j=0; j<= totalSize; j++){
                        FoutFile << j << " ";
                        for(int i=0; i<=n; i++){
                            FoutFile <<FIXED_FLOAT(foft[i][j]) << " ";
                        }
                        FoutFile << "\n";
                    }
                    for(int j=0; j< totalSize; j++){
                        if (verbalMode) cout << j << "\t" <<f0[j] << "\t\t\t" << fnow[j] << "\t\t\t" <<  "\n"; //<<  Soperator[j]<< "\t\t\t" << fnow[j] + Soperator[j] 
                        SoutFile << FIXED_FLOAT(Soperator[j]) << "\n";
                    }

                }else{
                   for (int j =0; j < totalSize; j++) {
                        if (verbalMode) cout << j << "\t" << f0[j] << "\t\t\t" << fnow[j] << "\t\t\t" <<  "\n"; //<<  Soperator[j]<< "\t\t\t" << fnow[j] + Soperator[j] 
                        SoutFile << j << " " << FIXED_FLOAT(Soperator[j]) <<  "\n";
                        FoutFile << j << " " << FIXED_FLOAT(fnow[j]) <<  "\n";
                    }
                }
			if (verbalMode) cout << totalSize << "\t" << Slength   <<  "\n";
				
			}else{
				cout << "Failed to open file for output file for Fout!\n";
			}
			FoutFile.close();
		}else{
			cout << "Failed to open output file for Smatrix!\n";
			return 0;
		}


		SoutFile.close();

				// Calculating total time taken by the program. 
		double time_taken = double(end - start) ; 
        uint64_t timing_ms = end_ms-start_ms;
        uint64_t timing_ns = end_ns-start_ns;
        if (verbalMode) {
		cout << "Time taken by program is: " << fixed 
			<< time_taken << setprecision(9); 
		cout << " sec " << endl; 
		cout << "Time taken by program is: " << fixed 
			<< timing_ms << setprecision(9); 
		cout << " micro sec " << endl; 
		cout << "Time taken by program is: " << fixed 
			<< timing_ns << setprecision(9); 
		cout << " nano sec " << endl; 
        }else{
            for(int j=0; j<nMax+1; j++)
                cout << aMax[j] << " ";
            cout <<", " << totalSize << ", " << deltaT << ", " << n << ", " << timing_ms << ", " << (double)timing_ms/(double)n << "; in microseconds\n";
        }
        
		
        return 0;
    }else{
        cout << "Couldn't open Finit file, check your preferences.\n Aborting Calculation\n";
        return 0;
    }    
    
}
