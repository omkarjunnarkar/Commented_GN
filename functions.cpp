/*
***********************************************************************************************************************************************************

												 Gauss-Newton Algorithm
												Fitting Routine Functions

Author: Omkar Junnarkar, Semester-3 MSc. Computational Material Science
Matriculation Nr.: 66157	Email: omkar.junnarkar@student.tu-freiberg.de
IDE : Microsoft Visual Studio 2019

iostream: For Input-Output of the C++ Program
Eigen/Dense: For Using Dense Matrix Interface of Linear Algebra Library Eigen
iomanip: To manipulate the number of decimal places in output
math: For arithmetic operations
fstream: To create the stream of file and write
functions.h: Contains the Fitting Routine Header
*/
#include<iostream>
#include<Eigen/Dense>
#include<iomanip>
#include<math.h>
#include<fstream>
#include"functions.h"

/*
To reduce the effort of specifying libraries/class for each command/data type
*/
using namespace std;
using namespace Eigen;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*
	Arguments : Parameters of Equation, Input(x) for Equation
	para( , ) : List of Parameters
	x( , ) : List of Input values for the equation

	Output : Matrix 'y', data points of the funcction y=f(parameters,x)
*/
MatrixXd function_y(MatrixXd para, MatrixXd x) {
	int number_of_data_points = x.rows();
	MatrixXd y(number_of_data_points, 1);
	for (int i = 0; i < number_of_data_points; i++) {

		y(i, 0) = para(0, 0) * pow(x(i, 0), 4) + para(1, 0) * exp(para(2, 0) * x(i, 0));
		//y(i, 0) = para(0, 0) * pow(x(i, 0), 3) + para(1, 0) * pow(x(i, 0), 2)*cos(para(2,0)*x(i,0));
		//y(i, 0) = para(0, 0) * pow(x(i, 0), 2) + para(1, 0) * exp(para(2, 0) * x(i, 0)) + para(3, 0) * x(i, 0);
	}
	return y;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*
	Jacobian Matrix : the matrix containing values of First Order Partial Derivatives w.r.t all parameters at All Data Points
	Size : (Number of Data Points, Parameters)
	The Partial Derivatives of the functions with respect to each Parameter can be estimated by using Finite Difference Method as follows :

	Df/Da = [f(a + DELa) - f(a)]/DELa , Df/Db = [f(b + DELb) - f(b)]/DELb , Df/Dc = [f(c + DELc) - f(c)]/DELc
	where a,b,c are paramerters and DELa,DELb,DELc are the deflections (given by initial deflection)

	Thus by computing function values of these derivates at all data points 'x', All Elements of Jacobian can be obtained.
	[ Refer Report/Manual for Details ]
	y_deflected : Function value obtained by deflecting parameter

	Arguments: Estimated Parameters, Õnitial Deflection, 'y' measured, Input data 'x'
	Output : Jacobian Matrix
*/
MatrixXd getJacobianMatrix(MatrixXd para_est, MatrixXd deflection, MatrixXd ym, MatrixXd input) {

	MatrixXd Jacobian_Matrix(ym.rows(), para_est.rows());
	MatrixXd y = function_y(para_est, input);
	MatrixXd y_deflected(ym.rows(), 1);

	for (int i = 0; i < para_est.rows(); i++) {

		para_est(i, 0) = para_est(i, 0) + deflection(i, 0);		/*Changing the parameters one by one */

		y_deflected = function_y(para_est, input);				/*Computing the deflected function arrray */
		for (int j = 0; j < input.rows(); j++) {

			// [f(v, p + dp) - f(v, p) ] / [dp] 

			Jacobian_Matrix(j, i) = (y_deflected(j, 0) - y(j, 0)) / deflection(i, 0);
		}
		para_est(i, 0) = para_est(i, 0) - deflection(i, 0);		/*Bringing back the parametes to original value*/
	}
	return Jacobian_Matrix;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*
	Gauss-Newton Algorithm  :

	Arguments : Estimated Parameters, Initial Deflection of Parameters, Measured 'y' values, Data Input 'x'
	Output : True Parameters

	Computation Strategy & Variables:

	-> Hessian Matrix (H) and difference between measured y values and estimated y values is computed (d) along with error
	-> Inverse of Hessian is computed. To avoid the necessity of a positive definite matrix, Psuedo Inverse is computed using Orthogonal Decomposition of the Matrix
	-> Change in Parameters (dp) is computed by H.PsuedoInverse*J.transpose*d -> Parameters are updated and new error is computed
	-> Iterations continue untill problem converges

	[ Refer Report/Manual for Details ]

*/

MatrixXd GaussNewton(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, MatrixXd input) {

	cout << "-> Entered Gauﬂ-Newton\n";

	ofstream errorfile("ErrorNorm.csv");
	ofstream hessianfile("Hessians.csv");
	
	/* Number of Parameters */
	int npara = para_guess.rows(), ndata = ym.rows();

	MatrixXd H(npara, npara);
	MatrixXd d(ndata, 1);
	MatrixXd J(ndata, npara);
	double error,error_gn;

	MatrixXd y_init = function_y(para_guess, input);
	
	MatrixXd para_est = para_guess;			/* Estimated Parameters*/
	int maxiter = 1000, counter=0;			/* Maximum Iterations Allowed, Counter for Iterations */
	

	while (counter < maxiter) {
		
		cout << "--> Iteration : " << counter << endl;

		J = getJacobianMatrix(para_est, deflection, ym,input);
		cout << "J: \n" << J << endl;
		MatrixXd y_est = function_y(para_est, input);
		d = ym - y_est;
		//cout << "d: \n" << d << endl;
		H = J.transpose() * J;
		cout << "H: \n" << H << endl;

		if (counter == 0) {					/* Only to be done for First Iteration */
			MatrixXd temp1 = d.transpose() * d;
			error = temp1(0,0);
			//cout << "error" << error << endl;
			}
		
		cout << "Hinverse=\n" << H.completeOrthogonalDecomposition().pseudoInverse() << endl;	/*Printing out the PsuedoInverse of Matrix*/
		MatrixXd dp = H.completeOrthogonalDecomposition().pseudoInverse() * J.transpose() * d;	/*Computing change in parameters*/
		//cout << "dp: \n" << dp;
		MatrixXd para_gn = para_est + dp;						/* Update of parameters*/
		MatrixXd y_est_gn= function_y(para_gn, input);			/* 'y' values from updated parameters*/
		MatrixXd d_gn = ym - y_est_gn;							/* difference betwen measured and computed values*/
		MatrixXd temp2 = d_gn.transpose() * d_gn;				/* Present Error */
		error_gn = temp2(0,0);
		
		para_est = para_gn;
		error = error_gn;

		/* Stopping Criterion */
		if (dp.norm() < 1e-4) {
			counter = maxiter;
		}
		else counter++;
		
		/* Writing Files */
		hessianfile << H << endl << endl;
		errorfile << error_gn << endl;
		
	}

	hessianfile.close();
	errorfile.close();

	/* Returning Final Set of Obtained Parameters */

	return para_est;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
