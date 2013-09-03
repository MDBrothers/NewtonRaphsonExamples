#include "NewtonRaphson.hpp"
#include <cstdlib>

void CalcDepVarsCS(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, std::complex<double> >& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated);

void CalcDepVarsAD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const std::vector<NRNameSpace::F>& currentGuesses, 
         			       std::vector<NRNameSpace::F>& targetsCalculated);

void CalcDepVarsFD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated);

void CalcDepVars(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated);

void CalcJacobian(const Teuchos::SerialDenseMatrix<int, double>& constants,
                                 Teuchos::SerialDenseMatrix<int, double>& jacobian,
                           const Teuchos::SerialDenseVector<int, double>& currentGuesses);

int main(int argc, char * argv[])
{
    Teuchos::SerialDenseVector<int, std::complex<double> >  yourInitialGuessCS(1); 
    Teuchos::SerialDenseVector<int, double>  yourInitialGuessAD(1); 
    Teuchos::SerialDenseVector<int, double>  yourInitialGuessFD(1); 
    Teuchos::SerialDenseVector<int, double>  yourTargetsDesired(1);
    Teuchos::SerialDenseMatrix<int, double>  yourConstants(1, 3);

    yourInitialGuessCS = std::complex<double>(2.0, 0.0);
    yourInitialGuessAD = 2.0; //First guess is that the end of the truss bar will deflect 1 centimeter
    yourInitialGuessFD = yourInitialGuessAD;
    yourTargetsDesired = 24.4484; //Load in Newtons
    yourConstants = 0.0;
    yourConstants(0,0) = 44483985.77; //Rigidity in Newtons
    yourConstants(0,1) = sqrt(pow(2.54,2.0) + pow(254.0,2.0)); //Original Length of Truss bar in centimeters
	yourConstants(0,2) = 10.51; //Constant of vertical spring in newtons per centimeter

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

    NRNameSpace::NRUserSupplied Watermelon(
            yourInitialGuessFD,
            yourTargetsDesired,
            yourConstants,
            CalcDepVars,
            CalcJacobian);

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
            Watermelon.solve(UPDATEMULTIPLIER, MAXITERATIONS, ERRORTOLLERANCE);
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
        std::cout << "and PROBELENGTH, UPDATEMULTIPLIER, in order, like this:" << std::endl;
        std::cout << "./examples.exe 20 1.E-7 1.E-10 .15" << std::endl;
    }


	return 0;
}

void CalcDepVarsCS(const Teuchos::SerialDenseMatrix<int, double>& constants, 
   		  			 const Teuchos::SerialDenseVector<int, std::complex<double> >& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated)
        {
            std::complex<double> Displacement = currentGuesses[0];
            std::complex<double> CosPhi = 254.0*pow(pow(-Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//A/H
            std::complex<double> SinPhi = (-Displacement + 2.54)*pow(pow(-Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//O/H

            double Rigidity = constants(0,0);
            double OriginalLength = constants(0,1);
            double kSpring = constants(0,2);

            targetsCalculated[0] = .5 * Rigidity * pow(CosPhi, 2.0) *  pow(Displacement*pow(OriginalLength, -1.0), 2.0)*
                                   (Displacement*pow(OriginalLength, -1.0)*pow(CosPhi, 2.0) - 3.0*SinPhi) +
                                   kSpring*Displacement + (Rigidity*Displacement*pow(OriginalLength, -1.0))*pow(SinPhi, 2.0);

           }

void CalcDepVarsAD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
                   const std::vector<NRNameSpace::F>& currentGuesses, 
         			       std::vector<NRNameSpace::F>& targetsCalculated)
        {
            NRNameSpace::F Displacement = currentGuesses[0];
            NRNameSpace::F CosPhi = 254.0*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//A/H
            NRNameSpace::F SinPhi = (-Displacement + 2.54)*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//O/H

            double Rigidity = constants(0,0);
            double OriginalLength = constants(0,1);
            double kSpring = constants(0,2);

            targetsCalculated[0] = .5 * Rigidity * pow(CosPhi, 2.0) *  pow(currentGuesses[0]*pow(OriginalLength, -1.0), 2.0)*
                                   (currentGuesses[0]*pow(OriginalLength, -1.0)*pow(CosPhi, 2.0) - 3.0*SinPhi) +
                                   kSpring*currentGuesses[0] + (Rigidity*currentGuesses[0]*pow(OriginalLength, -1.0))*pow(SinPhi, 2.0);

         }

void CalcDepVarsFD(const Teuchos::SerialDenseMatrix<int, double>& constants, 
                   const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated)
        {
            double Displacement = currentGuesses[0];
            double CosPhi = 254.0*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//A/H
            double SinPhi = (-Displacement + 2.54)*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//O/H

            double Rigidity = constants(0,0);
            double OriginalLength = constants(0,1);
            double kSpring = constants(0,2);

            targetsCalculated[0] = .5 * Rigidity * pow(CosPhi, 2.0) *  pow(Displacement*pow(OriginalLength, -1.0), 2.0)*
                                   (Displacement*pow(OriginalLength, -1.0)*pow(CosPhi, 2.0) - 3.0*SinPhi) +
                                   kSpring*Displacement + (Rigidity*Displacement*pow(OriginalLength, -1.0))*pow(SinPhi, 2.0);
        }

void CalcDepVars(const Teuchos::SerialDenseMatrix<int, double>& constants, 
                   const Teuchos::SerialDenseVector<int, double>& currentGuesses, 
         			       Teuchos::SerialDenseVector<int, double>& targetsCalculated)
        {
          //  std::cout << "currentGuesses: " << currentGuesses << std::endl;
            double Displacement = currentGuesses[0];
            double CosPhi = 254.0*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//A/H
            double SinPhi = (-Displacement + 2.54)*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//O/H

            double Rigidity = constants(0,0);
            double OriginalLength = constants(0,1);
            double kSpring = constants(0,2);

            targetsCalculated[0] = .5 * Rigidity * pow(CosPhi, 2.0) *  pow(Displacement*pow(OriginalLength, -1.0), 2.0)*
                                   (Displacement*pow(OriginalLength, -1.0)*pow(CosPhi, 2.0) - 3.0*SinPhi) +
                                   kSpring*Displacement + (Rigidity*Displacement*pow(OriginalLength, -1.0))*pow(SinPhi, 2.0);

        //    std::cout << "Targets calculated: " << targetsCalculated[0] << std::endl;
        }


void CalcJacobian(const Teuchos::SerialDenseMatrix<int, double>& constants, 
                            Teuchos::SerialDenseMatrix<int, double>& jacobian,
                           const Teuchos::SerialDenseVector<int, double>& currentGuesses)
        {
            double Displacement = currentGuesses[0];
            double CosPhi = 254.0*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//A/H
            double SinPhi = (-Displacement + 2.54)*pow(pow(-1.0*Displacement +2.54, 2.0) +  pow(254.0, 2.0), -.5);//O/H

            double Rigidity = constants(0,0);
            double OriginalLength = constants(0,1);
            double kSpring = constants(0,2);

            jacobian(0,0) = 1.5*Rigidity*pow(CosPhi,2.0)*
                            (Displacement*pow(OriginalLength, -1.0)*pow(CosPhi, 2.0) -2.0*SinPhi)*
                            Displacement*pow(OriginalLength, -2.0) + kSpring + Rigidity*pow(SinPhi,2.0)*pow(OriginalLength, -1.0); 
        }
