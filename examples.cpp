#include "NewtonRaphson.hpp"
#include <cstdlib>

const int NUMDIMENSIONS =3;

void CalcDepVarsCS(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, std::complex<double> >& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated);

void CalcDepVarsAD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const std::vector<NRNameSpace::F>& currentGuesses, 
         			       std::vector<NRNameSpace::F>& targetsCalculated);

void CalcDepVarsFD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated);

int main(int argc, char * argv[])
{
    Teuchos::SerialDenseVector<int, std::complex<double> >  yourInitialGuessCS(NUMDIMENSIONS); 
    Teuchos::SerialDenseVector<int, double>  yourInitialGuessAD(NUMDIMENSIONS); 
    Teuchos::SerialDenseVector<int, double>  yourInitialGuessFD(NUMDIMENSIONS); 
    Teuchos::SerialDenseVector<int, double>  yourTargetsDesired(NUMDIMENSIONS);
    Teuchos::SerialDenseMatrix<int, double>  yourConstants(NUMDIMENSIONS, NUMDIMENSIONS);

    yourInitialGuessCS = std::complex<double>(2.0, 0.0);
    yourInitialGuessAD = 2.0;
    yourInitialGuessFD = yourInitialGuessAD;

    yourTargetsDesired = 0.0;
    yourConstants = 0.0;
    yourConstants(0,0) = 1.0;
    yourConstants(1,2) = 1.0;
	yourConstants(2,2) = 1.0;

    NRNameSpace::NRComplexStep Apples(
            NUMDIMENSIONS,
            yourInitialGuessCS,
            yourTargetsDesired,
            yourConstants,
            CalcDepVarsCS);

    NRNameSpace::NRForwardDifference Oranges(
            NUMDIMENSIONS,
            yourInitialGuessFD,
            yourTargetsDesired,
            yourConstants,
            CalcDepVarsFD);

    NRNameSpace::NRAutomaticDifferentiation Bananas(
            NUMDIMENSIONS,
            yourInitialGuessAD,
            yourTargetsDesired,
            yourConstants,
            CalcDepVarsAD);

    bool correctARGS = true;
    if(argc == 4)
    {
        int MAXITERATIONS;
        double ERRORTOLLERANCE;
        double PROBELENGTH;

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

        if(correctARGS)
        {
            Apples.solve(MAXITERATIONS, ERRORTOLLERANCE, PROBELENGTH);
            std::cout << "############################################" << std::endl;
            Oranges.solve(MAXITERATIONS, ERRORTOLLERANCE, PROBELENGTH);
            std::cout << "############################################" << std::endl;
            Bananas.solve(MAXITERATIONS, ERRORTOLLERANCE);
        }
        else
            std::cout << "Argument values need to be interpretable as:" << std::endl;
            std::cout << "integer, double, double" << std::endl;

    }
    else
    {
        std::cout << "Call the executable again, but with numerical" << std::endl;
        std::cout << "arguments for MAXITERATIONS, ERRORTOLLERANCE," << std::endl;
        std::cout << "and PROBELENGTH, in order, like this:" << std::endl;
        std::cout << "./examples.exe 20 1.E-7 1.E-10" << std::endl;
    }


	return 0;
}

void CalcDepVarsCS(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, std::complex<double> >& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated)
        {
    	    //Evaluate a dependent variable for each iteration
	        for(int i = 0; i < NUMDIMENSIONS; i++)
	        {
	    	    targetsCalculated[i] = pow(currentGuesses[0] - constants(i, 0), 2.0);
	    	    targetsCalculated[i] += pow(currentGuesses[1] - constants(i,  1), 2.0);
	    	    targetsCalculated[i] += currentGuesses[2]*pow(-1.0,i) - constants(i,  2);
	        }
        }

void CalcDepVarsAD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const std::vector<NRNameSpace::F>& currentGuesses, 
         			       std::vector<NRNameSpace::F>& targetsCalculated)
        {
    	    //Evaluate a dependent variable for each iteration
	        for(int i = 0; i < NUMDIMENSIONS; i++)
	        {
	    	    targetsCalculated[i] = pow(currentGuesses[0] - constants(i, 0), 2.0);
	    	    targetsCalculated[i] += pow(currentGuesses[1] - constants(i,  1), 2.0);
	    	    targetsCalculated[i] += currentGuesses[2]*pow(-1.0,i) - constants(i,  2);
	        }
        }

void CalcDepVarsFD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated)
        {
    	    //Evaluate a dependent variable for each iteration
	        for(int i = 0; i < NUMDIMENSIONS; i++)
	        {
	    	    targetsCalculated[i] = pow(currentGuesses[0] - constants(i, 0), 2.0);
	    	    targetsCalculated[i] += pow(currentGuesses[1] - constants(i,  1), 2.0);
	    	    targetsCalculated[i] += currentGuesses[2]*pow(-1.0,i) - constants(i,  2);
	        }
        }

