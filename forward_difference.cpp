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
#include <Teuchos_RCP.hpp>
#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_BLAS.hpp>
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_Version.hpp"
#include <valarray>

const int NUMDIMENSIONS = 3;
const int MAXITERATIONS = 9;
const double ERRORTOLLERANCE = 1.0E-4;
const double PROBEDISTANCE = 1.E-12;

void calculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>& myOffsets,
				 const Teuchos::SerialDenseVector<int, double>& myCurrentGuess, 
		                 Teuchos::SerialDenseVector<int, double>& targetsCalculated);

void calculateJacobian(const Teuchos::SerialDenseMatrix<int, double>& myOffsets,
			Teuchos::SerialDenseMatrix<int, double>& myJacobian, 
		       Teuchos::SerialDenseVector<int, double>& myTargetsCalculated, 
		       Teuchos::SerialDenseVector<int, double>& myUnperturbedTargetsCalculated,
		       Teuchos::SerialDenseVector<int, double>& myCurrentGuess, 
		       void myCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, const Teuchos::SerialDenseVector<int, double>&, Teuchos::SerialDenseVector<int, double>&),
		       Teuchos::BLAS<int, double>& myBLAS);

void updateGuess(Teuchos::SerialDenseVector<int, double>& myCurrentGuess,
		Teuchos::SerialDenseVector<int, double>& myGuessAdjustment,
		Teuchos::SerialDenseVector<int, double>& myTargetsCalculated,
		Teuchos::SerialDenseMatrix<int, double>& myJacobian,
		Teuchos::LAPACK<int, double>& myLAPACK,
		Teuchos::SerialDenseSolver<int, double>& mySolver,
		Teuchos::BLAS<int, double>& myBLAS); 

void calculateResidual(const Teuchos::SerialDenseVector<int, double>& myTargetsDesired, 
		       const Teuchos::SerialDenseVector<int, double>& myTargetsCalculated,
		       Teuchos::BLAS<int, double>& myBLAS,
		       double& myError);

int main(int argc, char* argv[])
{
	//--This first declaration is included for software engineering reasons: allow main method to control flow of data
	//create function pointer for calculateDependentVariable
	//We want to do this so that calculateJacobian can call calculateDependentVariables as it needs to,
	//but we explicitly give it this authority from the main method
	void (*yourCalculateDependentVariables)(const Teuchos::SerialDenseMatrix<int, double>&, const Teuchos::SerialDenseVector<int, double>&, Teuchos::SerialDenseVector<int, double>&);
	yourCalculateDependentVariables = &calculateDependentVariables;

	//Create an object to allow access to LAPACK using a library found in Trilinos
	//
	Teuchos::LAPACK<int, double> Teuchos_LAPACK_Object;	
	Teuchos::SerialDenseSolver<int, double> solver;
	Teuchos::BLAS<int, double> Teuchos_BLAS_Object;

	//The problem being solved is to find the intersection of three infinite paraboloids:
	//(x-1)^2 + y^2 + z = 0
	//x^2 + y^2 -(z+1) = 0
	//x^2 + y^2 +(z-1) = 0
	//
	//They should intersect at the point (1, 0, 0)
	Teuchos::SerialDenseMatrix<int, double> offsets(NUMDIMENSIONS, NUMDIMENSIONS); 
	offsets(0,0) = 1.0;
	offsets(1,2) = 1.0;
	offsets(2,2) = 1.0;

	//We need to initialize the target vectors and provide an initial guess
	Teuchos::SerialDenseVector<int, double> targetsDesired(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, double> targetsCalculated(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, double> unperturbedTargetsCalculated(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, double> currentGuess(NUMDIMENSIONS);
	Teuchos::SerialDenseVector<int, double> guessAdjustment(NUMDIMENSIONS);

	//Place to store our tangent-stiffness matrix or Jacobian
	//One element for every combination of dependent variable with independent variable
	Teuchos::SerialDenseMatrix<int, double> jacobian(NUMDIMENSIONS, NUMDIMENSIONS);

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
                                  yourCalculateDependentVariables,
				  Teuchos_BLAS_Object);


		//Compute a guessChange and immediately set the currentGuess equal to the guessChange
 		updateGuess(currentGuess,
			    targetsCalculated,
			    guessAdjustment,
			    jacobian,
			    Teuchos_LAPACK_Object,
			    solver,
			    Teuchos_BLAS_Object);


		//Compute F(x) with the updated, currentGuess
		calculateDependentVariables(offsets,
				            currentGuess,
			       		    targetsCalculated);	


		//Calculate the L2 norm of Ftarget - F(xCurrentGuess)
		calculateResidual(targetsDesired,
			          targetsCalculated,
				  Teuchos_BLAS_Object,
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
void calculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>& myOffsets,
				 const Teuchos::SerialDenseVector<int, double>& myCurrentGuess, 
		                 Teuchos::SerialDenseVector<int, double>& targetsCalculated)
{
	//Evaluate a dependent variable for each iteration
	for(int i = 0; i < NUMDIMENSIONS; i++)
	{
		targetsCalculated[i] = pow(myCurrentGuess[0] - myOffsets(i, 0), 2.0);
		targetsCalculated[i] += pow(myCurrentGuess[1] - myOffsets(i,  1), 2.0);
		targetsCalculated[i] += myCurrentGuess[2]*pow(-1.0,i) - myOffsets(i,  2);
	}
	
}

void calculateJacobian(const Teuchos::SerialDenseMatrix<int, double>& myOffsets,
		       Teuchos::SerialDenseMatrix<int, double>& myJacobian, 
		       Teuchos::SerialDenseVector<int, double>& myTargetsCalculated, 
		       Teuchos::SerialDenseVector<int, double>& myUnperturbedTargetsCalculated,
		       Teuchos::SerialDenseVector<int, double>& myCurrentGuess, 
		       void myCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, const Teuchos::SerialDenseVector<int, double>&, Teuchos::SerialDenseVector<int, double>&),
			Teuchos::BLAS<int, double>& myBLAS)
{

	//Calculate a temporary, unperturbed target evaluation, such as is needed for the finite-difference
	//formula
	myCalculateDependentVariables(myOffsets, myCurrentGuess, myUnperturbedTargetsCalculated);
	double oldGuessValue = 0.0;

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
		myCurrentGuess[column] += PROBEDISTANCE;

		//Evaluate functions for perturbed guess
		myCalculateDependentVariables(myOffsets, myCurrentGuess, myTargetsCalculated);

			
		//Do forward difference: subtraction 
		myBLAS.AXPY(NUMDIMENSIONS, -1.0, myUnperturbedTargetsCalculated.values(), myUnperturbedTargetsCalculated.stride(), myTargetsCalculated.values(), myTargetsCalculated.stride());

		//Do forward difference: divide by step size
		myBLAS.SCAL(NUMDIMENSIONS, pow(PROBEDISTANCE, -1.0), myTargetsCalculated.values(), myTargetsCalculated.stride());

		for(int row = 0; row < NUMDIMENSIONS; row ++)
		{
			myJacobian(row, column) = myTargetsCalculated[row];
		}

		myCurrentGuess[column] = oldGuessValue;
	}

	//Reset to unperturbed, so we dont waste a function evaluation
	myBLAS.COPY(NUMDIMENSIONS, myTargetsCalculated.values(), myTargetsCalculated.stride(), myUnperturbedTargetsCalculated.values(), myUnperturbedTargetsCalculated.stride());

}

void updateGuess(Teuchos::SerialDenseVector<int, double>& myCurrentGuess,
		Teuchos::SerialDenseVector<int, double>& myGuessAdjustment,
		Teuchos::SerialDenseVector<int, double>& myTargetsCalculated,
		Teuchos::SerialDenseMatrix<int, double>& myJacobian, 
		Teuchos::LAPACK<int, double>& myLAPACK,
		Teuchos::SerialDenseSolver<int, double>& mySolver,
		Teuchos::BLAS<int, double>& myBLAS
		 )
{
	//v = J(inverse) * (-F(x))
	//new guess = v + old guess

	myBLAS.SCAL(NUMDIMENSIONS, -1.0, myTargetsCalculated.values(), myTargetsCalculated.stride());
	//Perform an LU factorization of this matrix. 
	int ipiv[NUMDIMENSIONS], info;
	char TRANS = 'N';
//	myLAPACK.GETRF( NUMDIMENSIONS, NUMDIMENSIONS, myJacobian.values(), myJacobian.stride(), ipiv, &info ); 
	// Solve the linear system.
//	myLAPACK.GETRS( TRANS, NUMDIMENSIONS, 1, myJacobian.values(), myJacobian.stride(), ipiv, myTargetsCalculated.values(), myTargetsCalculated.stride(), &info );  
	//We have overwritten myTargetsCalculated with guess update values
	Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > A_ptr = Teuchos::rcpFromRef(myJacobian);
	Teuchos::RCP<Teuchos::SerialDenseVector<int, double> > x_ptr = Teuchos::rcpFromRef(myGuessAdjustment);
	Teuchos::RCP<Teuchos::SerialDenseVector<int, double> > b_ptr = Teuchos::rcpFromRef(myTargetsCalculated);
	mySolver.setMatrix(A_ptr);
	mySolver.setVectors(x_ptr, b_ptr);
	mySolver.solve();
	
	myBLAS.AXPY(NUMDIMENSIONS, 1.0, myGuessAdjustment.values(), myGuessAdjustment.stride(), myCurrentGuess.values(), myCurrentGuess.stride());

}

void calculateResidual(const Teuchos::SerialDenseVector<int, double>& myTargetsDesired, 
		       const Teuchos::SerialDenseVector<int, double>& myTargetsCalculated,
		       Teuchos::BLAS<int, double>& myBLAS,
		       double& myError)
{
	//error is the l2 norm of the difference from my state to my target
	myBLAS.AXPY(NUMDIMENSIONS, -1.0,  myTargetsDesired.values(), myTargetsDesired.stride(), myTargetsCalculated.values(), myTargetsCalculated.stride());


        myError = myBLAS.NRM2(NUMDIMENSIONS, myTargetsCalculated.values(), myTargetsCalculated.stride());
}
