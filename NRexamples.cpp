#include "NewtonRaphson.hpp"
#include <cstdlib>

void CalcDepVarsCS(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, std::complex<double> >& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated,
                     const Teuchos::SerialDenseVector<int, double>& targetsDesired);

void CalcDepVarsAD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const std::vector<NRNameSpace::F>& currentGuesses, 
         			       std::vector<NRNameSpace::F>& targetsCalculated,
                     const Teuchos::SerialDenseVector<int, double>& targetsDesired);

void CalcDepVarsFD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated,
                     const Teuchos::SerialDenseVector<int, double>& targetsDesired);

int main(int argc, char * argv[])
{
    Teuchos::SerialDenseVector<int, std::complex<double> >  yourInitialGuessCS(3); 
    Teuchos::SerialDenseVector<int, double>  yourInitialGuessAD(3); 
    Teuchos::SerialDenseVector<int, double>  yourInitialGuessFD(3); 
    Teuchos::SerialDenseVector<int, double>  yourTargetsDesired(3);
    Teuchos::SerialDenseMatrix<int, double>  yourConstants(3, 3);

    yourInitialGuessCS = std::complex<double>(2.0, 0.0);
    yourInitialGuessAD = 2.0;
    yourInitialGuessFD = yourInitialGuessAD;

    yourTargetsDesired = 0.0;
    yourConstants = 0.0;
    yourConstants(0,0) = 1.0;
    yourConstants(1,2) = 1.0;
	yourConstants(2,2) = 1.0;

    NRNameSpace::NRComplexStep Apples(
            yourInitialGuessCS,
            yourTargetsDesired,
            yourConstants,
            CalcDepVarsCS);

    NRNameSpace::NRForwardDifference Oranges(
            yourInitialGuessFD,
            yourTargetsDesired,
            yourConstants,
            CalcDepVarsFD);

    NRNameSpace::NRAutomaticDifferentiation Bananas(
            yourInitialGuessAD,
            yourTargetsDesired,
            yourConstants,
            CalcDepVarsAD);

    bool correctARGS = true;
    if(argc == 5)
    {
        int MAXITERATIONS;
        double ERRORTOLLERANCE;
        double PROBELENGTH;
        double UPDATEMULTIPLIER;

        if(MAXITERATIONS = atoi(argv[1]))
        {
            std::cout << "Value selected for MAXITERATIONS is a valid integer." << std::endl;
        }
        else
            correctARGS = false;

        if(ERRORTOLLERANCE = atof(argv[2]))
        {
            std::cout << "Value selected for ERRORTOLLERANCE is a valid double." << std::endl;
        }
        else 
            correctARGS = false;

        if(PROBELENGTH = atof(argv[3]))
        {
            std::cout << "Value selected for PROBELENGTH is a valid double." << std::endl;
        }
        else
            correctARGS = false;

        if(UPDATEMULTIPLIER = atof(argv[4]))
        {
            std::cout << "Value selected for UPDATEMULTIPLIER is a valid double." << std::endl;
        }
        else
            correctARGS = false;


        if(correctARGS)
        {
            Apples.solve(UPDATEMULTIPLIER, MAXITERATIONS, ERRORTOLLERANCE, PROBELENGTH);
            Oranges.solve(UPDATEMULTIPLIER, MAXITERATIONS, ERRORTOLLERANCE, PROBELENGTH);
            Bananas.solve(UPDATEMULTIPLIER, MAXITERATIONS, ERRORTOLLERANCE);
        }
        else
        {
            std::cout << "Argument values need to be interpretable as:" << std::endl;
            std::cout << "integer, double, double, double" << std::endl;
        }

    }
    else
    {
        std::cout << "Call the executable again, but with numerical" << std::endl;
        std::cout << "arguments for MAXITERATIONS, ERRORTOLLERANCE," << std::endl;
        std::cout << "and PROBELENGTH, UPDATEMULTIPLIER in order, like this:" << std::endl;
        std::cout << "./examples.exe 20 1.E-7 1.E-10 .15" << std::endl;
    }


	return 0;
}

void CalcDepVarsCS(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, std::complex<double> >& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated,
                    const Teuchos::SerialDenseVector<int, double>& targetsDesired )
        {
    	    //Evaluate a dependent variable for each iteration
	        for(int i = 0; i < targetsCalculated.length(); i++)
	        {
	    	    targetsCalculated[i] = pow(currentGuesses[0] - constants(i, 0), 2.0);
	    	    targetsCalculated[i] += pow(currentGuesses[1] - constants(i,  1), 2.0);
	    	    targetsCalculated[i] += currentGuesses[2]*pow(-1.0,i) - constants(i,  2);
                targetsCalculated[i] -= targetsDesired[i];
	        }
        }

void CalcDepVarsAD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const std::vector<NRNameSpace::F>& currentGuesses, 
         			       std::vector<NRNameSpace::F>& targetsCalculated,
                     const Teuchos::SerialDenseVector<int, double>& targetsDesired)
        {
    	    //Evaluate a dependent variable for each iteration
	        for(int i = 0; i < targetsCalculated.size(); i++)
	        {
	    	    targetsCalculated[i] = pow(currentGuesses[0] - constants(i, 0), 2.0);
	    	    targetsCalculated[i] += pow(currentGuesses[1] - constants(i,  1), 2.0);
	    	    targetsCalculated[i] += currentGuesses[2]*pow(-1.0,i) - constants(i,  2);
                targetsCalculated[i] -= targetsDesired[i];
	        }
        }

void CalcDepVarsFD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated,
                   const Teuchos::SerialDenseVector<int, double>& targetsDesired)
        {
    	    //Evaluate a dependent variable for each iteration
	        for(int i = 0; i < targetsCalculated.length(); i++)
	        {
	    	    targetsCalculated[i] = pow(currentGuesses[0] - constants(i, 0), 2.0);
	    	    targetsCalculated[i] += pow(currentGuesses[1] - constants(i,  1), 2.0);
	    	    targetsCalculated[i] += currentGuesses[2]*pow(-1.0,i) - constants(i,  2);
                targetsCalculated[i] -= targetsDesired[i];
	        }
        }

