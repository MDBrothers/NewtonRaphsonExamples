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

The method "calculateDependentVariables" is specific to this problem, however everything else is largely general.

The method "calculateJacobian" demonstrates the forward-difference technique:
1)Unperturbed model evaluation is computed 
2)The model is re-evaluated 
3)The change from unperturbed to perturbed over probe distance is an approximation of a set of first partial derivatives
w.r.t the perturbed dof
4)The calculated derivatives are a column of the Jacobian matrix.

The Jacobian matrix looks like this:
dE1/dx | dE1/dy | dE1/dz
------------------------
dE2/dx | dE2/dy | dE2/dz
------------------------
dE3/dx | dE3/dy | dE3/dz

The Newton Raphson scheme works like this:
1)Guess a x,y,z to satisfy all equations
2)Evaluate all equations and store their results
3)At the location of your guess point x,y,z compute a tangent-stiffness matrix (Jacobian)
4)Don't do 4, but generally: guess_new = guess_current + inverse_of_Jacobian * -1.0 * equations_results_from_guess_current
5)guess_new - guess_current = update_ammount; Jacobian * update_amount = -1.0 * equations_results_from_guess_current
6)Therefor, solve for update_amount using 5 rather than invert the Jacobian. This is much faster.
7)Loop back to step 1 and carry out subsequent steps until equations_results_from_guess_current is close to your target as desired
--or-- the maximum number of iterations allowable has been exceeded according to your criteria.

####Dependencies:
The "Armadillo" C++ API was chosen to handle the linear algebra tasks because of its clear syntax
and integration with LAPACK. For a list of dependences and installation instructions for
Armadillo refer to the Armadillo project page at sourceforge.net.

As of 20 Aug. 2013, Armadillo can be obtained at: http://arma.sourceforge.net/
*/

#include <iostream>
#include <armadillo>

const int NUMDIMENSIONS = 3;
const int MAXITERATIONS = 9;
const double ERRORTOLLERANCE = 1.0E-4;
//If probe distance is made too small, subtractive cancellation may occur, leading to inaccurate derivatives
//recompile with a small value for PROBEDISTANCE, like 1.0E-30 to see the effect
const double PROBEDISTANCE = 1.0E-10;

void calculateDependentVariables(const arma::Mat<double>& myOffsets,
				 const arma::Col<double>& myCurrentGuess, 
		                 arma::Col<double>& targetsCalculated);

void calculateJacobian(const arma::Mat<double>& myOffsets,
			arma::Mat<double>& myJacobian, 
		       arma::Col<double>& myTargetsCalculated, 
		       arma::Col<double>& myCurrentGuess, 
		       void myCalculateDependentVariables(const arma::Mat<double>&, const arma::Col<double>&, arma::Col<double>&));

void updateGuess(arma::Col<double>& myCurrentGuess,
		 const arma::Col<double>& myTargetsCalculated,
		 const arma::Mat<double>& myJacobian);

void calculateResidual(const arma::Col<double>& myTargetsDesired, 
		       const arma::Col<double>& myTargetsCalculated,
		       double& myError);

int main(int argc, char* argv[])
{
	//--This first declaration is included for software engineering reasons: allow main method to control flow of data
	//create function pointer for calculateDependentVariable
	//We want to do this so that calculateJacobian can call calculateDependentVariables as it needs to,
	//but we explicitly give it this authority from the main method
	void (*yourCalculateDependentVariables)(const arma::Mat<double>&, const arma::Col<double>&, arma::Col<double>&);
	yourCalculateDependentVariables = &calculateDependentVariables;

	//The problem being solved is to find the intersection of three infinite paraboloids:
	//(x-1)^2 + y^2 + z = 0
	//x^2 + y^2 -(z+1) = 0
	//x^2 + y^2 +(z-1) = 0
	//
	//They should intersect at the point (1, 0, 0)
	arma::Mat<double> offsets(NUMDIMENSIONS, NUMDIMENSIONS); 
	offsets.fill(0.0);
	offsets.col(0)[0] = 1.0;
	offsets.col(2)[1] = 1.0;
	offsets.col(2)[2] = 1.0;

	//We need to initialize the target vectors and provide an initial guess
	arma::Col<double> targetsDesired(NUMDIMENSIONS);
	targetsDesired.fill(0.0);

	arma::Col<double> targetsCalculated(NUMDIMENSIONS);
	targetsCalculated.fill(0.0);

	arma::Col<double> currentGuess(NUMDIMENSIONS);
	currentGuess.fill(2.0);

	arma::Col<double> guessChange(NUMDIMENSIONS);
	guessChange.fill(0.0);

	//Place to store our tangent-stiffness matrix or Jacobian
	//One element for every combination of dependent variable with independent variable
	arma::Mat<double> jacobian(NUMDIMENSIONS, NUMDIMENSIONS);
	jacobian.fill(0.0);

	int count = 0;
	double error = 1.0E5;

	std::cout << "Running forward difference example ..........." << std::endl;
	while(count < MAXITERATIONS and error > ERRORTOLLERANCE)
	{

		//Calculate Jacobian tangent to currentGuess point
		//at the same time, an unperturbed targetsCalculated is, well, calculated
		calculateJacobian(offsets,
				  jacobian,
				  targetsCalculated,
				  currentGuess,
                                  yourCalculateDependentVariables);

		//Compute a new currentGuess
		updateGuess(currentGuess,
			    targetsCalculated,
			    jacobian);

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
	std::cout << "Final guess:\nx, y, z\n " << currentGuess.t();
	std::cout << "Error tollerance: " << ERRORTOLLERANCE << std::endl;
	std::cout << "Final error: " << error << std::endl;
	std::cout << "--program complete--" << std::endl;

	return 0;
}


//This function is specific to a single problem
void calculateDependentVariables(const arma::Mat<double>& myOffsets,
				 const arma::Col<double>& myCurrentGuess, 
		                 arma::Col<double>& targetsCalculated)
{
	//Evaluate a dependent variable for each iteration
	//The arma::Col allows this to be expressed as a vector operation
	for(int i = 0; i < NUMDIMENSIONS; i++)
	{
		targetsCalculated[i] = arma::sum(pow(myCurrentGuess.subvec(0,1) - myOffsets.row(i).subvec(0,1).t(),2.0));
		targetsCalculated[i] = targetsCalculated[i] + myCurrentGuess[2]*pow(-1.0, i) - myOffsets.row(i)[2]; 
		//std::cout << targetsCalculated[i] << std::endl;
	}
	//std::cout << "model evaluated *************************" << std::endl;
	//std::cout << targetsCalculated << std::endl;
	//std::cout << myOffsets << std::endl;
	
}

void calculateJacobian(const arma::Mat<double>& myOffsets,
		       arma::Mat<double>& myJacobian, 
		       arma::Col<double>& myTargetsCalculated, 
		       arma::Col<double>& myCurrentGuess, 
		       void myCalculateDependentVariables(const arma::Mat<double>&, const arma::Col<double>&, arma::Col<double>&))
{
	//Calculate a temporary, unperturbed target evaluation, such as is needed for the finite-difference
	//formula
	arma::Col<double> unperturbedTargetsCalculated(NUMDIMENSIONS);
	unperturbedTargetsCalculated.fill(0.0);
	myCalculateDependentVariables(myOffsets, myCurrentGuess, unperturbedTargetsCalculated);
	double oldGuessValue = 0.0;

	//Each iteration fills a column in the Jacobian
	//The Jacobian takes this form:
	//
	//	dF0/dx0 dF0/dx1 
	//	dF1/dx0 dF1/dx1
	//
	for(int j = 0; j< NUMDIMENSIONS; j++)
	{
		//Store old element value, perturb the current value
		oldGuessValue = myCurrentGuess[j];
		myCurrentGuess[j] += PROBEDISTANCE;

		//Evaluate functions for perturbed guess
		myCalculateDependentVariables(myOffsets, myCurrentGuess, myTargetsCalculated);

		//The column of the Jacobian that goes with the independent variable we perturbed
		//can be determined using the finite-difference formula
		//The arma::Col allows this to be expressed as a single vector operation
		//note slice works as: std::slice(start_index, number_of_elements_to_access, index_interval_between_selections)
		//std::cout << "Jacobian column " << j << " with:" << std::endl;
		//std::cout << "myTargetsCalculated" << std::endl;
		//std::cout << myTargetsCalculated << std::endl;
		//std::cout << "unperturbedTargetsCalculated" << std::endl;
		//std::cout << unperturbedTargetsCalculated << std::endl;
		myJacobian.col(j) = (myTargetsCalculated - unperturbedTargetsCalculated) * pow(PROBEDISTANCE, -1.0);
		//std::cout << "The jacobian: " << std::endl;
		//std::cout << myJacobian << std::endl;

		myCurrentGuess[j] = oldGuessValue;
	}

	//Reset to unperturbed, so we dont waste a function evaluation
	myTargetsCalculated = unperturbedTargetsCalculated;
}

void updateGuess(arma::Col<double>& myCurrentGuess,
		 const arma::Col<double>& myTargetsCalculated,
		 const arma::Mat<double>& myJacobian)
{
	//v = J(inverse) * (-F(x))
	//new guess = v + old guess
	std::cout << "Current Jacobian: " << std::endl;
	std::cout << myJacobian << std::endl;
	myCurrentGuess = myCurrentGuess + solve(myJacobian, -myTargetsCalculated, true);
}

void calculateResidual(const arma::Col<double>& myTargetsDesired, 
		       const arma::Col<double>& myTargetsCalculated,
		       double& myError)
{
	//error is the l2 norm of the difference from my state to my target
	myError = arma::norm((myTargetsDesired - myTargetsCalculated), 2);
}


