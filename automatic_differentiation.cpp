/*
####Title: 
Example Newton Raphson Solver: Automatic Differentiation 

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
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_LAPACK.hpp>
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_Version.hpp"
#include <Sacado.hpp>
#include <valarray>

typedef Sacado::Fad::DFad<double>  F;  // Forward AD with # of ind. vars given later

const int NUMDIMENSIONS = 3;
const int MAXITERATIONS = 9;
const double ERRORTOLLERANCE = 1.0E-4;
//No probing is done with AD as in the other methods

void calculateDependentVariables(const std::valarray<F>& myOffsets,
				 const std::valarray<F>& myCurrentGuess, 
		                 std::valarray<F>& targetsCalculated);

void calculateJacobian(const std::valarray<F>& myOffsets,
			Teuchos::SerialDenseMatrix<int, double>& myJacobian, 
		       std::valarray<F>& myTargetsCalculated, 
		       std::valarray<F>& myCurrentGuess, 
		       void myCalculateDependentVariables(const std::valarray<F>&, const std::valarray<F>&, std::valarray<F>&));

void updateGuess(std::valarray<F>& myCurrentGuess,
		Teuchos::SerialDenseVector<int, double>& myTargetsCalculatedValuesOnly,
		 const std::valarray<F>& myTargetsCalculated,
		Teuchos::SerialDenseMatrix<int, double>& myJacobian,
		Teuchos::LAPACK<int, double>& myLAPACK); 

void calculateResidual(const std::valarray<F>& myTargetsDesired, 
		       const std::valarray<F>& myTargetsCalculated,
		       double& myError);

int main(int argc, char* argv[])
{
	//--This first declaration is included for software engineering reasons: allow main method to control flow of data
	//create function pointer for calculateDependentVariable
	//We want to do this so that calculateJacobian can call calculateDependentVariables as it needs to,
	//but we explicitly give it this authority from the main method
	void (*yourCalculateDependentVariables)(const std::valarray<F>&, const std::valarray<F>&, std::valarray<F>&);
	yourCalculateDependentVariables = &calculateDependentVariables;

	//Create an object to allow access to LAPACK using a library found in Trilinos
	//
	Teuchos::LAPACK<int, double> Teuchos_LAPACK_Object;	

	//The problem being solved is to find the intersection of three infinite paraboloids:
	//(x-1)^2 + y^2 + z = 0
	//x^2 + y^2 -(z+1) = 0
	//x^2 + y^2 +(z-1) = 0
	//
	//They should intersect at the point (1, 0, 0)
	std::valarray<F> offsets(0.0, NUMDIMENSIONS*NUMDIMENSIONS); 
	offsets[0] = 1.0;
	offsets[2*NUMDIMENSIONS -1] = 1.0;
	offsets[3*NUMDIMENSIONS -1] = 1.0;

	//We need to initialize the target vectors and provide an initial guess
	std::valarray<F> targetsDesired(0.0, NUMDIMENSIONS);

	std::valarray<F> targetsCalculated(0.0, NUMDIMENSIONS);

	Teuchos::SerialDenseVector<int, double> targetsCalculatedValuesOnly(NUMDIMENSIONS);

	std::valarray<F> currentGuess(2.0, NUMDIMENSIONS);
	//designate the elements of the currentGuess vector as independent variables that we want to take partial derivatives
	//with respect to later on in the program.
	for(int i = 0; i < NUMDIMENSIONS; i++)
	{
		//inependentVariable.diff(degree_of_freedom_assignment, total_number_of_degrees_of_freedom);
		currentGuess[i].diff(i, NUMDIMENSIONS);
	}

	//Place to store our tangent-stiffness matrix or Jacobian
	//One element for every combination of dependent variable with independent variable
	//Column major
	Teuchos::SerialDenseMatrix<int, double> jacobian(NUMDIMENSIONS, NUMDIMENSIONS);

	int count = 0;
	double error = 1.0E5;

	std::cout << "Running automatic differentiation example ................" << std::endl;
	while(count < MAXITERATIONS and error > ERRORTOLLERANCE)
	{

		//Calculate Jacobian tangent to currentGuess point
		//at the same time, an unperturbed targetsCalculated is, well, calculated
		calculateJacobian(offsets,
				  jacobian,
				  targetsCalculated,
				  currentGuess,
                                  yourCalculateDependentVariables);

		//Compute a guessChange and immediately set the currentGuess equal to the guessChange
 		updateGuess(currentGuess,
			    targetsCalculatedValuesOnly,
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
	std::cout << "Final guess:\n x, y, z\n " << currentGuess[0].val() << ", " << currentGuess[1].val() << ", " << currentGuess[2].val() << std::endl;
	std::cout << "Error tollerance: " << ERRORTOLLERANCE << std::endl;
	std::cout << "Final error: " << error << std::endl;
	std::cout << "--program complete--" << std::endl;

	return 0;
}


//This function is specific to a single problem
void calculateDependentVariables(const std::valarray<F>& myOffsets,
				 const std::valarray<F>& myCurrentGuess, 
		                 std::valarray<F>& targetsCalculated)
{
	//Evaluate a dependent variable for each iteration
	//The std::valarray<F> allows this to be expressed as a vector operation
	for(int i = 0; i < NUMDIMENSIONS; i++)
	{
		targetsCalculated[i] = pow(myCurrentGuess[std::slice(0,2,1)] - myOffsets[std::slice(i*NUMDIMENSIONS, 2, 1)],2.0).sum();
		targetsCalculated[i] = targetsCalculated[i] + myCurrentGuess[2]*pow(-1.0, i) - myOffsets[i*NUMDIMENSIONS + 2]; 
		//std::cout << targetsCalculated[i] << std::endl;
	}
	//std::cout << "model evaluated *************************" << std::endl;
	//std::cout << targetsCalculated << std::endl;
	//std::cout << myOffsets << std::endl;
	
}

void calculateJacobian(const std::valarray<F>& myOffsets,
			Teuchos::SerialDenseMatrix<int, double>& myJacobian, 
		       std::valarray<F>& myTargetsCalculated, 
		       std::valarray<F>& myCurrentGuess, 
		       void myCalculateDependentVariables(const std::valarray<F>&, const std::valarray<F>&, std::valarray<F>&))
{
	//evaluate the model only once
	myCalculateDependentVariables(myOffsets, myCurrentGuess, myTargetsCalculated);

	//Each iteration fills a column in the Jacobian
	//The Jacobian takes this form:
	//
	//	dF0/dx0 dF0/dx1 
	//	dF1/dx0 dF1/dx1
	//
	for(int j = 0; j< NUMDIMENSIONS; j++)
	{

		myCalculateDependentVariables(myOffsets, myCurrentGuess, myTargetsCalculated);

		//extract the derivatives computed for us by the AD system
		for(int i = 0; i < NUMDIMENSIONS; i++)
		{
			myJacobian(i,j) = myTargetsCalculated[i].dx(j);
		}

	}
}

void updateGuess(std::valarray<F>& myCurrentGuess,
		Teuchos::SerialDenseVector<int, double>& myTargetsCalculatedValuesOnly,
		 const std::valarray<F>& myTargetsCalculated,
		Teuchos::SerialDenseMatrix<int, double>& myJacobian, 
		Teuchos::LAPACK<int, double>& myLAPACK
		 )
{
	//v = J(inverse) * (-F(x))
	//new guess = v + old guess
	for(int i = 0; i < NUMDIMENSIONS; i++)
	{
		myTargetsCalculatedValuesOnly[i] = -myTargetsCalculated[i].val();
	}

	//Perform an LU factorization of this matrix. 
	int ipiv[NUMDIMENSIONS], info;
	char TRANS = 'N';
	myLAPACK.GETRF( NUMDIMENSIONS, NUMDIMENSIONS, myJacobian.values(), myJacobian.stride(), ipiv, &info ); 
	    
	// Solve the linear system.
	myLAPACK.GETRS( TRANS, NUMDIMENSIONS, 1, myJacobian.values(), myJacobian.stride(), 
	ipiv, myTargetsCalculatedValuesOnly.values(), myTargetsCalculatedValuesOnly.stride(), &info );  

	//We have overwritten myTargetsCalculatedValuesOnly with guess update values
	for(int i = 0; i < NUMDIMENSIONS; i++)
	{
		myCurrentGuess[i] += myTargetsCalculatedValuesOnly[i];
	}
}

void calculateResidual(const std::valarray<F>& myTargetsDesired, 
		       const std::valarray<F>& myTargetsCalculated,
		       double& myError)
{
	//error is the l2 norm of the difference from my state to my target
	myError = pow(pow((myTargetsDesired - myTargetsCalculated), 2.0).sum(), .5).val();
}


