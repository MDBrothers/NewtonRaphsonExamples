/*
####Title: 
Example Newton Raphson Solver: Forward Difference

####Author: 
Michael D. Brothers

####Affiliation: 
University of Texas at San Antonio

####Date: 
20 Aug. 2013

####Notes: 
Program solves for the location of the sole intersection of three infinite paraboloids
which exist as parabola shaped surfaces in 3D space according to the following equations:

(x-1)^2 + y^2 + z = 0
x^2 + y^2 -(z+1) = 0
x^2 + y^2 +(z-1) = 0

They should intersect at the point (1, 0, 0)
*/

#include <iostream>
#include <complex>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_BLAS.hpp>
#include <Teuchos_LAPACK.hpp>
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_Version.hpp"

const int NUMDIMENSIONS = 3;
const int MAXITERATIONS = 9;
const double ERRORTOLLERANCE = 1.0E-4;
const double PROBEDISTANCE = 1.E-19;

void calculateDependentVariables(const Teuchos::SerialDenseMatrix<int, std::complex<double> >& myOffsets,
				 const Teuchos::SerialDenseVector<int, std::complex<double> >& myCurrentGuess, 
		                 Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated);

void calculateJacobian(const Teuchos::SerialDenseMatrix<int, std::complex<double> >& myOffsets,
			Teuchos::SerialDenseMatrix<int, std::complex<double> >& myJacobian, 
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsCalculated, 
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myUnperturbedTargetsCalculated,
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myCurrentGuess, 
		       void myCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, std::complex<double> >&, const Teuchos::SerialDenseVector<int, std::complex<double> >&, Teuchos::SerialDenseVector<int, std::complex<double> >&));

void updateGuess(Teuchos::SerialDenseVector<int, std::complex<double> >& myCurrentGuess,
		Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsCalculated,
		Teuchos::SerialDenseMatrix<int, std::complex<double> >& myJacobian,
		Teuchos::LAPACK<int, std::complex<double> >& myLAPACK);

void calculateResidual(const Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsDesired, 
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsCalculated,
		       double& myError);

int main(int argc, char* argv[])
{
	//--This first declaration is included for software engineering reasons: allow main method to control flow of data
	//create function pointer for calculateDependentVariable
	//We want to do this so that calculateJacobian can call calculateDependentVariables as it needs to,
	//but we explicitly give it this authority from the main method
	void (*yourCalculateDependentVariables)(const Teuchos::SerialDenseMatrix<int, std::complex<double> >&, const Teuchos::SerialDenseVector<int, std::complex<double> >&, Teuchos::SerialDenseVector<int, std::complex<double> >&);
	yourCalculateDependentVariables = &calculateDependentVariables;

	//Create an object to allow access to LAPACK using a library found in Trilinos
	//
	Teuchos::LAPACK<int, std::complex<double> > Teuchos_LAPACK_Object;	

	//The problem being solved is to find the intersection of three infinite paraboloids:
	//(x-1)^2 + y^2 + z = 0
	//x^2 + y^2 -(z+1) = 0
	//x^2 + y^2 +(z-1) = 0
	//
	//They should intersect at the point (1, 0, 0)
	Teuchos::SerialDenseMatrix<int, std::complex<double> > offsets(NUMDIMENSIONS, NUMDIMENSIONS); 
	offsets(0,0) = 1.0;
	offsets(1,2) = 1.0;
	offsets(2,2) = 1.0;

	//We need to initialize the target vectors and provide an initial guess
	Teuchos::SerialDenseVector<int, std::complex<double> > targetsDesired(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, std::complex<double> > targetsCalculated(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, std::complex<double> > unperturbedTargetsCalculated(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, std::complex<double> > currentGuess(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, std::complex<double> > guessAdjustment(NUMDIMENSIONS);

	currentGuess[0] = 2.0;
	currentGuess[1] = 1.0;
	currentGuess[2] = 1.0;
	//Place to store our tangent-stiffness matrix or Jacobian
	//One element for every combination of dependent variable with independent variable
	Teuchos::SerialDenseMatrix<int, std::complex<double> > jacobian(NUMDIMENSIONS, NUMDIMENSIONS);

	int count = 0;
	double error = 1.0E5;

	std::cout << "Running Forward Difference Example" << std::endl;
	while(count < MAXITERATIONS and error > ERRORTOLLERANCE)
	{


		//Calculate Jacobian tangent to currentGuess point
		//at the same time, an unperturbed targetsCalculated is, well, calculated
		calculateJacobian(offsets,
				  jacobian,
				  targetsCalculated,
				  unperturbedTargetsCalculated,
				  currentGuess,
                                  yourCalculateDependentVariables);


		//Compute a guessChange and immediately set the currentGuess equal to the guessChange
 		updateGuess(currentGuess,
			    targetsCalculated,
			    jacobian,
			    Teuchos_LAPACK_Object);


		//Compute F(x) with the updated, currentGuess
		calculateDependentVariables(offsets,
				            currentGuess,
			       		    targetsCalculated);	


		//Calculate the L2 norm of Ftarget - F(xCurrentGuess)
		calculateResidual(targetsDesired,
			          targetsCalculated,
			  	  error);	  


		count ++;
		//If we have converged, or if we have exceeded our alloted number of iterations, discontinue the loop
		std::cout << "Residual Error: " << error << std::endl;
	}
	
	std::cout << "******************************************" << std::endl;
	std::cout << "Number of iterations: " << count << std::endl;
	std::cout << "Final guess:\n x, y, z\n " << currentGuess[0] << ", " << currentGuess[1] << ", " << currentGuess[2] << std::endl;
	std::cout << "Error tollerance: " << ERRORTOLLERANCE << std::endl;
	std::cout << "Final error: " << error << std::endl;
	std::cout << "--program complete--" << std::endl;

	return 0;
}

//This function is specific to a single problem
void calculateDependentVariables(const Teuchos::SerialDenseMatrix<int, std::complex<double> >& myOffsets,
				 const Teuchos::SerialDenseVector<int, std::complex<double> >& myCurrentGuess, 
		                 Teuchos::SerialDenseVector<int, std::complex<double> >& targetsCalculated)
{
	//Evaluate a dependent variable for each iteration
	for(int i = 0; i < NUMDIMENSIONS; i++)
	{
		targetsCalculated[i] = pow(myCurrentGuess[0] - myOffsets(i, 0), 2.0);
		targetsCalculated[i] += pow(myCurrentGuess[1] - myOffsets(i,  1), 2.0);
		targetsCalculated[i] += myCurrentGuess[2]*pow(-1.0,i) - myOffsets(i,  2);
	}
	
}

void calculateJacobian(const Teuchos::SerialDenseMatrix<int, std::complex<double> >& myOffsets,
		       Teuchos::SerialDenseMatrix<int, std::complex<double> >& myJacobian, 
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsCalculated, 
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myUnperturbedTargetsCalculated,
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myCurrentGuess, 
		       void myCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, std::complex<double> >&, const Teuchos::SerialDenseVector<int, std::complex<double> >&, Teuchos::SerialDenseVector<int, std::complex<double> >&)
		)
{

	//Calculate a temporary, unperturbed target evaluation, such as is needed for the finite-difference
	//formula
	myCalculateDependentVariables(myOffsets, myCurrentGuess, myUnperturbedTargetsCalculated);
	std::complex<double> oldGuessValue;

	//Each iteration fills a column in the Jacobian
	//The Jacobian takes this form:
	//
	//	dF0/dx0 dF0/dx1 
	//	dF1/dx0 dF1/dx1
	//
	//
	for(int column = 0; column< NUMDIMENSIONS; column++)
	{
		//Store old element value, perturb the current value
		oldGuessValue = myCurrentGuess[column];
		myCurrentGuess[column] += std::complex<double>(0.0, PROBEDISTANCE);

		//Evaluate functions for perturbed guess
		myCalculateDependentVariables(myOffsets, myCurrentGuess, myTargetsCalculated);

		myTargetsCalculated *= pow(PROBEDISTANCE, -1.0);

		for(int row = 0; row < NUMDIMENSIONS; row ++)
		{
			myJacobian(row, column) = std::imag(myTargetsCalculated[row]);
		}

		myCurrentGuess[column] = oldGuessValue;
	}

	//Reset to unperturbed, so we dont waste a function evaluation
	
	myTargetsCalculated = myUnperturbedTargetsCalculated;
}

void updateGuess(Teuchos::SerialDenseVector<int, std::complex<double> >& myCurrentGuess,
		Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsCalculated,
		Teuchos::SerialDenseMatrix<int, std::complex<double> >& myJacobian, 
		Teuchos::LAPACK<int, std::complex<double> >& myLAPACK
		 )
{
	//v = J(inverse) * (-F(x))
	//new guess = v + old guess
	myTargetsCalculated *= -1.0;

	//Perform an LU factorization of this matrix. 
	int ipiv[NUMDIMENSIONS], info;
	char TRANS = 'N';
	myLAPACK.GETRF( NUMDIMENSIONS, NUMDIMENSIONS, myJacobian.values(), myJacobian.stride(), ipiv, &info ); 

	// Solve the linear system.
	myLAPACK.GETRS( TRANS, NUMDIMENSIONS, 1, myJacobian.values(), myJacobian.stride(),
		       	ipiv, myTargetsCalculated.values(), myTargetsCalculated.stride(), &info );  

	//We have overwritten myTargetsCalculated with guess update values
	//myBLAS.AXPY(NUMDIMENSIONS, 1.0, myGuessAdjustment.values(), 1, myCurrentGuess.values(), 1);
	myCurrentGuess += myTargetsCalculated;
}

void calculateResidual(const Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsDesired, 
		       Teuchos::SerialDenseVector<int, std::complex<double> >& myTargetsCalculated,
		       double& myError)
{
	//error is the l2 norm of the difference from my state to my target
	myTargetsCalculated -= myTargetsDesired;
	std::complex<double> intermediate = myTargetsCalculated.dot(myTargetsCalculated);
	myError = std::real(sqrt(intermediate));
}
