#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

// #include "./QmatrixPA4_unrotated.cpp"
// #include "./QmatrixPA6_unrotated.cpp"
#include "./QmatrixPA8_unrotated.cpp"

struct S
{
    double coefficient;
    int caseNumber;
    int startIndexResult;
    int startIndexLeft;
    int startIndexRight;

    S(double coeff, int caseN, int startIRes, int startIL, int startIR) : coefficient(coeff), caseNumber(caseN), startIndexResult(startIRes), startIndexLeft(startIL), startIndexRight(startIR) {}
};

void readSinit(std::ifstream &Sinit, int *&aMax, int &nMax)
{
    int init[2] = {0, 0};
    std::string line;
    getline(Sinit, line);
    std::stringstream nvalue(line);
    nvalue >> nMax;

    aMax = new int[nMax];
    getline(Sinit, line);
    std::stringstream aValues(line);
    for (int i = 0; i < nMax; i++)
    {
        aValues >> aMax[i];
    }
    nMax--;
}

int main(int argc, char **argv)
{
    std::string locationSinit;
    std::string locationSvalues;
    std::string locationFinit;
    std::string locationSout;
    std::string locationFout;
    int *aMax, nMax, Slength = 0;
    std::vector<S> Smatrix;
    if (argc < 2)
    {
        std::cout << "Please state a preferences file\n";
        return 0;
    }

    // get the file locations for the Smatrix
    std::ifstream preferences(argv[1]);
    if (preferences.is_open())
    {
        getline(preferences, locationSinit);
        getline(preferences, locationSvalues);
        getline(preferences, locationFinit);
        getline(preferences, locationSout);
        getline(preferences, locationFout);
        preferences.close();
    }
    else
    {
        std::cout << "Couldn't open preferences file.\n Aborting Calculation.\n";
        return 0;
    }

    // read out the initialization parameters aMax, nMax, Slength from the Sinit-file
    std::ifstream Sinit(locationSinit);
    if (Sinit.is_open())
    {
        readSinit(Sinit, aMax, nMax);
        Sinit.close();
    }
    else
    {
        std::cout << "Couldn't open S-Matrix initialization file, check your preferences.\nAborting Calculation.\n";
        return 0;
    }

    // initialize the vector for the SMatrix: Scoefficients (which contains all the physical information) and Scases (which maps the coefficients to the correct Q and start indexes of f and fprime)
    std::ifstream Smatrixfile(locationSvalues);
    if (Smatrixfile.is_open())
    {
        std::string line;
        double coeff;
        int para[4] = {0, 0, 0, 0};
        while (getline(Smatrixfile, line))
        {
            std::stringstream smvalues(line);
            smvalues >> coeff >> para[0] >> para[1] >> para[2] >> para[3];
            Smatrix.push_back(S(coeff, para[0], para[1], para[2], para[3]));
            Slength++;
        }
        Smatrixfile.close();
        // the last element of Smatrix is added twice, removing it now and setting Slength to the correct value
        Slength--;
        Smatrix.pop_back();
    }
    else
    {
        std::cout << "Couldn't open S-Matrix file, check your preferences.\n Aborting Calculation\n";
        return 0;
    }

    std::ifstream Finitfile(locationFinit);
    if (Finitfile.is_open())
    {
        // initialize the moment vector and the collision operator as a one-dimensional array with a particular order of the parameters - similar to [n][a][l]

        std::vector<double> f0;  // f0 in moments
        std::string line;
        double coeff; // these are moments
        int idx;

        // initialize the values of fnow as given in the Finitfile
        while (getline(Finitfile, line))
        {
            std::stringstream finitvalues(line);
            finitvalues >> idx >> coeff;
            f0.push_back(coeff); // we are pushing moments into the f0 vector
        }
        Finitfile.close();

        std::vector<double> Soperator(f0.size(), 0);
        //Soperator.resize(f0.size());

        // Calculate collision
        for (int j = 0; j < Slength; j++)
        {
            calcQ(Smatrix[j].caseNumber, &f0[Smatrix[j].startIndexLeft], &f0[Smatrix[j].startIndexRight], &Soperator[Smatrix[j].startIndexResult], Smatrix[j].coefficient);
        }

        // Print collided vector
        for (int i = 0; i < Soperator.size(); i++)
        {
            std::cout << i << " " << Soperator[i] << std::endl;
        }
    }
}
