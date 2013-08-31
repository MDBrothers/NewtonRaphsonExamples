#include <iostream>
#include <Teuchos_BLAS.hpp>
#include <Teuchos_LAPACK.hpp>
#include <complex>
#include <vector>
#include <Sacado.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_SerialDenseSolver.hpp>
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_Version.hpp"

namespace NRNameSpace{

    typedef Sacado::Fad::DFad<double>  F;  // Forward AD with # of ind. vars given later

	class NewtonRaphsonProblem 
	{
		public:
			int NUMDIMENSIONS;
			double myError;

			Teuchos::SerialDenseMatrix<int, double> myJacobian;
			Teuchos::SerialDenseMatrix<int, double> myConstants;
			Teuchos::SerialDenseVector<int, double> myScratchReal;
			Teuchos::SerialDenseVector<int, double> myTargetsDesired;
			Teuchos::LAPACK<int, double> LAPACK;

			NewtonRaphsonProblem(){};
	};

	class NRForwardDifference: private NewtonRaphsonProblem 
	{
		private:
			Teuchos::SerialDenseVector<int, double > myCurrentGuesses;
			Teuchos::SerialDenseVector<int, double > myTargetsCalculated;

	        void (*myCalculateDependentVariables)(const Teuchos::SerialDenseMatrix<int, double>&, 
									                const Teuchos::SerialDenseVector<int, double >&, 
									                Teuchos::SerialDenseVector<int, double >&);            

            void calculateJacobian(const double PROBELENGTH,
                            const Teuchos::SerialDenseMatrix<int, double>& constants,
                            Teuchos::SerialDenseMatrix<int,  double>& jacobian, 
                            Teuchos::SerialDenseVector<int, double >& currentGuesses, 
                            Teuchos::SerialDenseVector<int, double >& targetsCalculated,
                            Teuchos::SerialDenseVector<int, double >& unperturbedTargets,
                       void myCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, 
                                          const Teuchos::SerialDenseVector<int, double >&, 
                                                Teuchos::SerialDenseVector<int, double >&))
            {
                //evaluate model with no perturbation
                myCalculateDependentVariables(constants, currentGuesses, unperturbedTargets);
                double oldGuessValue = 0.0;

                for(int column = 0; column < NUMDIMENSIONS; column++)
                {
                    //Store old guess vector element value, perturb the current value
                    oldGuessValue = currentGuesses[column];
                    currentGuesses[column] += PROBELENGTH;

                    //Re-evaluate and apply forward difference formula 
                    myCalculateDependentVariables(constants, currentGuesses, targetsCalculated);
                    targetsCalculated -= unperturbedTargets;
                    targetsCalculated *= pow(PROBELENGTH, -1.0);

                    for(int row = 0; row < NUMDIMENSIONS; row ++)
                    {
                        jacobian(row, column) = targetsCalculated[row];
                    }
                    currentGuesses[column] = oldGuessValue;
                }
                //Reset to unperturbed, so we dont waste a function evaluation
                targetsCalculated = unperturbedTargets;
            }

            void updateGuess(Teuchos::SerialDenseVector<int, double >& currentGuesses,
                    Teuchos::SerialDenseVector<int, double >& targetsCalculated,
                    Teuchos::SerialDenseMatrix<int, double>& jacobian, 
                    Teuchos::LAPACK<int, double>& lapack)
            {
                //v = J(inverse) * (-F(x))
                //new guess = v + old guess
                targetsCalculated *= -1.0;

                //Perform an LU factorization of this matrix. 
                int ipiv[NUMDIMENSIONS], info;
                char TRANS = 'N';
                lapack.GETRF( NUMDIMENSIONS, NUMDIMENSIONS, jacobian.values(), jacobian.stride(), ipiv, &info ); 

                // Solve the linear system.
                lapack.GETRS( TRANS, NUMDIMENSIONS, 1, jacobian.values(), jacobian.stride(),
                            ipiv, targetsCalculated.values(), targetsCalculated.stride(), &info );  

                //We have overwritten targetsCalculated with guess update values
                //Now update current guesses
                currentGuesses += targetsCalculated;
            }

            void calculateResidual(const Teuchos::SerialDenseVector<int, double>& targetsDesired, 
                                   const Teuchos::SerialDenseVector<int, double >& targetsCalculated,
                            Teuchos::SerialDenseVector<int, double>& scratch,
                            double& error)
            {
                //error is the l2 norm of the difference from my current results to my desired
                //results 
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    scratch[i] = targetsCalculated[i] - targetsDesired[i];
                }
                error = sqrt(scratch.dot(scratch));
            }

		public:
			NRForwardDifference(const int MYNUMDIMENSIONS,
					    const Teuchos::SerialDenseVector<int, double >& initialGuess,
					    const Teuchos::SerialDenseVector<int, double>& targetsDesired,
					    const Teuchos::SerialDenseMatrix<int, double>& constants,
					    void yourCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, 
									       const Teuchos::SerialDenseVector<int, double >&, 
									       Teuchos::SerialDenseVector<int, double >&))
			{
                NUMDIMENSIONS = MYNUMDIMENSIONS;
                myCurrentGuesses.resize(NUMDIMENSIONS);
                myTargetsDesired.resize(NUMDIMENSIONS);
                myTargetsCalculated.resize(NUMDIMENSIONS);
                myScratchReal.resize(NUMDIMENSIONS);
                myJacobian.shapeUninitialized(NUMDIMENSIONS, NUMDIMENSIONS);
                myConstants.shapeUninitialized(NUMDIMENSIONS, NUMDIMENSIONS);

                myCurrentGuesses = initialGuess;
                myTargetsDesired = targetsDesired;
                myConstants = constants;
	            myCalculateDependentVariables = yourCalculateDependentVariables;
			};

            void solve(const int MAXITERATIONS, const double ERRORTOLLERANCE, const double PROBELENGTH)
            {
                int count = 0;
                double error = 1.0E5;

                std::cout << "******************************************" << std::endl;
                std::cout << "Forward Difference Example" << std::endl;

                while(count < MAXITERATIONS and error > ERRORTOLLERANCE)
                {
                    calculateJacobian(
                            PROBELENGTH,
                            myConstants,
                              myJacobian,
                              myCurrentGuesses,
                              myTargetsCalculated,
                              myScratchReal,
                              myCalculateDependentVariables);

                    updateGuess(myCurrentGuesses,
                             myTargetsCalculated,
                            myJacobian,
                            LAPACK);

                    calculateResidual(myTargetsDesired,
                                  myTargetsCalculated,
                                  myScratchReal,
                              error);	  

                    count ++;
                    std::cout << "Residual Error: " << error << std::endl;
                }
                
                std::cout << "******************************************" << std::endl;
                std::cout << "Number of iterations: " << count << std::endl;
                std::cout << "Final guess:\n x, y, z\n " << myCurrentGuesses[0] << ", " << myCurrentGuesses[1] << ", " << myCurrentGuesses[2] << std::endl;
                std::cout << "Error tollerance: " << ERRORTOLLERANCE << std::endl;
                std::cout << "Final error: " << error << std::endl;
            }
	};


	class NRComplexStep: private NewtonRaphsonProblem 
	{
		private:
			Teuchos::SerialDenseVector<int, std::complex<double> >  myScratchComplex;
			Teuchos::SerialDenseVector<int, std::complex<double> > myCurrentGuesses;
			Teuchos::SerialDenseVector<int, std::complex<double> > myTargetsCalculated;

	        void (*myCalculateDependentVariables)(const Teuchos::SerialDenseMatrix<int, double>&, 
									                const Teuchos::SerialDenseVector<int, std::complex<double> >&, 
									                Teuchos::SerialDenseVector<int, std::complex<double> >&);            

            void calculateJacobian(const double PROBELENGTH,
                            const Teuchos::SerialDenseMatrix<int, double>& constants,
                            Teuchos::SerialDenseMatrix<int,  double>& jacobian, 
                            Teuchos::SerialDenseVector<int, std::complex< double> >& currentGuesses, 
                            Teuchos::SerialDenseVector<int, std::complex< double> >& targetsCalculated,
                            Teuchos::SerialDenseVector<int, std::complex< double> >& unperturbedTargets,
                       void myCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, 
                                          const Teuchos::SerialDenseVector<int, std::complex< double> >&, 
                                                Teuchos::SerialDenseVector<int, std::complex< double> >&))
            {
                //evaluate the unperturbed model
                myCalculateDependentVariables(constants, currentGuesses, unperturbedTargets);
                std::complex< double> oldGuessValue;

                for(int column = 0; column < NUMDIMENSIONS; column++)
                {
                    //Store old element value, perturb the current value
                    oldGuessValue = currentGuesses[column];
                    currentGuesses[column] += std::complex< double>(0.0, PROBELENGTH);

                    //Evaluate model for perturbed guess
                    myCalculateDependentVariables(constants, currentGuesses, targetsCalculated);

                    //Divide each element by PROBELENGTH
                    targetsCalculated *= pow(PROBELENGTH, -1.0);

                    for(int row = 0; row < NUMDIMENSIONS; row ++)
                    {
                        //Complete expressing the CTSE formula and take the magnitudes of
                        //the imaginary parts of targetsCalculated as derivatives wrt, the
                        //row = dof.
                        jacobian(row, column) = std::imag(targetsCalculated[row]);
                    }
                    currentGuesses[column] = oldGuessValue;
                }
                //Reset to unperturbed, so we dont waste a function evaluation
                targetsCalculated = unperturbedTargets;
            }

            void updateGuess(Teuchos::SerialDenseVector<int, std::complex< double> >& currentGuesses,
                    Teuchos::SerialDenseVector<int, std::complex< double> >& targetsCalculated,
                    Teuchos::SerialDenseVector<int, double>& scratch,
                    Teuchos::SerialDenseMatrix<int, double>& jacobian, 
                    Teuchos::LAPACK<int, double>& lapack)
            {
                //v = J(inverse) * (-F(x))
                //new guess = v + old guess
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    scratch[i] = -std::real(targetsCalculated[i]);
                }

                //Perform an LU factorization of this matrix. 
                int ipiv[NUMDIMENSIONS], info;
                char TRANS = 'N';
                lapack.GETRF( NUMDIMENSIONS, NUMDIMENSIONS, jacobian.values(), jacobian.stride(), ipiv, &info ); 

                // Solve the linear system.
                lapack.GETRS( TRANS, NUMDIMENSIONS, 1, jacobian.values(), jacobian.stride(),
                            ipiv, scratch.values(), scratch.stride(), &info );  

                //We have overwritten targetsCalculated with guess update values
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    currentGuesses[i] += scratch[i];
                }
            }

            void calculateResidual(const Teuchos::SerialDenseVector<int, double>& targetsDesired, 
                                   const Teuchos::SerialDenseVector<int, std::complex< double> >& targetsCalculated,
                            Teuchos::SerialDenseVector<int, double>& scratch,
                            double& error)
            {
                //error is the l2 norm of the difference from my state to my target
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    scratch[i] = std::real(targetsCalculated[i]) - targetsDesired[i];
                }
                error = sqrt(scratch.dot(scratch));
            }
            
		public:
			NRComplexStep(const int MYNUMDIMENSIONS,
					    const Teuchos::SerialDenseVector<int, std::complex<double> >& initialGuess,
					    const Teuchos::SerialDenseVector<int, double>& targetsDesired,
					    const Teuchos::SerialDenseMatrix<int, double>& constants,
					    void yourCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, 
									       const Teuchos::SerialDenseVector<int, std::complex<double> >&, 
									       Teuchos::SerialDenseVector<int, std::complex<double> >&))
			{
                NUMDIMENSIONS = MYNUMDIMENSIONS;
                myCurrentGuesses.resize(NUMDIMENSIONS);
                myTargetsDesired.resize(NUMDIMENSIONS);
                myTargetsCalculated.resize(NUMDIMENSIONS);
                myScratchReal.resize(NUMDIMENSIONS);
                myScratchComplex.resize(NUMDIMENSIONS);
                myJacobian.shapeUninitialized(NUMDIMENSIONS, NUMDIMENSIONS);
                myConstants.shapeUninitialized(NUMDIMENSIONS, NUMDIMENSIONS);

                myCurrentGuesses = initialGuess;
                myTargetsDesired = targetsDesired;
                myConstants = constants;
	            myCalculateDependentVariables = yourCalculateDependentVariables;

			};

            void solve(const int MAXITERATIONS, const double ERRORTOLLERANCE, const double PROBELENGTH)
            {
                int count = 0;
                double error = 1.0E5;


                std::cout << "******************************************" << std::endl;
                std::cout << "Complex Step Example" << std::endl;
                while(count < MAXITERATIONS and error > ERRORTOLLERANCE)
                {
                    calculateJacobian(
                            PROBELENGTH,
                            myConstants,
                              myJacobian,
                              myCurrentGuesses,
                              myTargetsCalculated,
                              myScratchComplex,
                              myCalculateDependentVariables);

                    updateGuess(myCurrentGuesses,
                             myTargetsCalculated,
                             myScratchReal,
                            myJacobian,
                            LAPACK);

                    calculateResidual(myTargetsDesired,
                                  myTargetsCalculated,
                                  myScratchReal,
                              error);	  


                    count ++;
                    std::cout << "Residual Error: " << error << std::endl;
                }
                
                std::cout << "******************************************" << std::endl;
                std::cout << "Number of iterations: " << count << std::endl;
                std::cout << "Final guess:\n x, y, z\n " << std::real(myCurrentGuesses[0]) << ", " << std::real(myCurrentGuesses[1]) << ", " << std::real(myCurrentGuesses[2]) << std::endl;
                std::cout << "Error tollerance: " << ERRORTOLLERANCE << std::endl;
                std::cout << "Final error: " << error << std::endl;
            }
	};

	class NRAutomaticDifferentiation: private NewtonRaphsonProblem 
	{

		private:

            std::vector<F> myCurrentGuesses;
            std::vector<F> myTargetsCalculated;

	        void (*myCalculateDependentVariables)(const Teuchos::SerialDenseMatrix<int, double>&, 
									                const std::vector<F>&, 
									                std::vector<F>&);            

            void calculateJacobian(
                            const Teuchos::SerialDenseMatrix<int, double>& constants,
                            Teuchos::SerialDenseMatrix<int,  double>& jacobian, 
                            std::vector<F>& currentGuesses, 
                            std::vector<F>& targetsCalculated,
                       void myCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, 
                                          const std::vector<F>&, 
                                                std::vector<F>&))
            {
                myCalculateDependentVariables(constants, currentGuesses, targetsCalculated);

                for(int column = 0; column < NUMDIMENSIONS; column++)
                {
                    for(int row = 0; row < NUMDIMENSIONS; row ++)
                    {
                        jacobian(row, column) = targetsCalculated[row].dx(column);
                    }
                }
            }

            void updateGuess(std::vector<F>& currentGuesses,
                    std::vector<F>& targetsCalculated,
                    Teuchos::SerialDenseVector<int, double>& scratch,
                    Teuchos::SerialDenseMatrix<int, double>& jacobian, 
                    Teuchos::LAPACK<int, double>& lapack)
            {
                //v = J(inverse) * (-F(x))
                //new guess = v + old guess
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    scratch[i] = -targetsCalculated[i].val();
                }

                //Perform an LU factorization of this matrix. 
                int ipiv[NUMDIMENSIONS], info;
                char TRANS = 'N';
                lapack.GETRF( NUMDIMENSIONS, NUMDIMENSIONS, jacobian.values(), jacobian.stride(), ipiv, &info ); 

                // Solve the linear system.
                lapack.GETRS( TRANS, NUMDIMENSIONS, 1, jacobian.values(), jacobian.stride(),
                            ipiv, scratch.values(), scratch.stride(), &info );  

                //We have overwritten scratch 
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    currentGuesses[i] += scratch[i];
                }
            }

            void calculateResidual(const Teuchos::SerialDenseVector<int, double>& targetsDesired, 
                                   std::vector<F>& targetsCalculated,
                            Teuchos::SerialDenseVector<int, double>& scratch,
                            double& error)
            {
                //error is the l2 norm of the difference from my state to my target
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    scratch[i] = targetsCalculated[i].val() - targetsDesired[i];
                }
                error = sqrt(scratch.dot(scratch));
            }

            
		public:
			NRAutomaticDifferentiation(const int MYNUMDIMENSIONS,
					    const Teuchos::SerialDenseVector<int, double >& initialGuess,
					    const Teuchos::SerialDenseVector<int, double>& targetsDesired,
					    const Teuchos::SerialDenseMatrix<int, double>& constants,
					    void yourCalculateDependentVariables(const Teuchos::SerialDenseMatrix<int, double>&, 
									       const std::vector<F>&, 
									       std::vector<F>&))
			{
                NUMDIMENSIONS = MYNUMDIMENSIONS;
                myTargetsDesired.resize(NUMDIMENSIONS);
                myScratchReal.resize(NUMDIMENSIONS);
                myJacobian.shapeUninitialized(NUMDIMENSIONS, NUMDIMENSIONS);
                myConstants.shapeUninitialized(NUMDIMENSIONS, NUMDIMENSIONS);

                myCurrentGuesses.resize(NUMDIMENSIONS);
                myTargetsCalculated.resize(NUMDIMENSIONS);
                for(int i = 0; i < NUMDIMENSIONS; i++)
                {
                    myCurrentGuesses[i] =initialGuess[i];
                    myCurrentGuesses[i].diff(i, NUMDIMENSIONS);
                }

                myTargetsDesired = targetsDesired;
                myConstants = constants;
	            myCalculateDependentVariables = yourCalculateDependentVariables;
			};

            void solve(const int MAXITERATIONS, const double ERRORTOLLERANCE)
            {
                int count = 0;
                double error = 1.0E5;

                std::cout << "******************************************" << std::endl;
                std::cout << "Automatic Differentiation Example" << std::endl;
                while(count < MAXITERATIONS and error > ERRORTOLLERANCE)
                {
                    calculateJacobian(
                            myConstants,
                              myJacobian,
                              myCurrentGuesses,
                              myTargetsCalculated,
                              myCalculateDependentVariables);

                    updateGuess(myCurrentGuesses,
                             myTargetsCalculated,
                             myScratchReal,
                            myJacobian,
                            LAPACK);

                    calculateResidual(myTargetsDesired,
                                  myTargetsCalculated,
                                  myScratchReal,
                              error);	  

                    count ++;
                    //If we have converged, or if we have exceeded our alloted number of iterations, discontinue the loop
                    std::cout << "Residual Error: " << error << std::endl;
                }
                std::cout << "******************************************" << std::endl;
                std::cout << "Number of iterations: " << count << std::endl;
                std::cout << "Final guess:\n x, y, z\n " << myCurrentGuesses[0].val() << ", " << myCurrentGuesses[1].val() << ", " << myCurrentGuesses[2].val() << std::endl;
                std::cout << "Error tollerance: " << ERRORTOLLERANCE << std::endl;
                std::cout << "Final error: " << error << std::endl;
            }
	};
}

