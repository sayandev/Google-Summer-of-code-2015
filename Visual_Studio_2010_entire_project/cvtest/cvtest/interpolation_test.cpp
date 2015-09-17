// Include Opencv
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "ap.h"
#include <math.h>
#include "alglibinternal.h"
#include "alglibmisc.h"
#include "linalg.h"
#include "statistics.h"
#include "dataanalysis.h"
#include "specialfunctions.h"
#include "solvers.h"
#include "optimization.h"
#include "diffequations.h"
#include "fasttransforms.h"
#include "integration.h"
#include "interpolation.h"
#include <vector>
#include <numeric> 
#include <algorithm>
#include <iterator>
#include <armadillo>
#include <cstddef>      // std::size_t
#include <cmath>        // std::atan2
#include <valarray> 
#include <complex>
#include <boost/range.hpp>
#include <boost/foreach.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>

#include "OsiClp/OsiClpSolverInterface.hpp"
#include "CbcModel.hpp"
#include "CoinModel.hpp"

#ifdef COIN_USE_CLP
#include "OsiClpSolverInterface.hpp"
typedef OsiClpSolverInterface OsiXxxSolverInterface;
#endif

#ifdef COIN_USE_OSL
#include "OsiOslSolverInterface.hpp"
typedef OsiOslSolverInterface OsiXxxSolverInterface;
#include "ekk_c_api.h"
#endif

#ifdef COIN_USE_CPX
#include "OsiCpxSolverInterface.hpp"
typedef OsiCpxSolverInterface OsiXxxSolverInterface;
#endif

#include "CoinPackedVector.hpp"
#include "CoinPackedMatrix.hpp"

BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)

const double PI = 3.141592653589793238460;
using namespace boost::geometry;
typedef std::complex<double> Complex_fft;
typedef std::valarray<Complex_fft> CArray_fft;
namespace bg = boost::geometry;
typedef bg::model::d2::point_xy<double> point_2d;
typedef bg::model::polygon<point_2d> polygon_2d;
typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;

using namespace alglib;
using boost::geometry::get;


// C++ IO
#include <iostream>

// Namespaces
using namespace cv;
using namespace std;
using namespace arma;

std::vector<double> linspace(double first, double last, int len) {
  std::vector<double> result(len);
  double step = (last-first) / (len - 1);
  for (int i=0; i<len; i++) { result[i] = first + i*step; }
  return result;
}

std::vector<double> linspace_intrvl(double first, double last, int intrvl) {
  std::vector<double> result;
  double step = intrvl;
  double temp=first;
  double i=0;
  while (temp+step<last) 
  { 
		temp=first + i*step; 
		result.push_back(temp);
		i=i+1.0;
  }
  return result;
}

// Cooley–Tukey FFT (in-place) "rosettacode.org"
void fft1(CArray_fft& x)
{
    const size_t N = x.size();
    if (N <= 1) return;
 
    // divide
    CArray_fft even = x[std::slice(0, N/2, 2)];
    CArray_fft  odd = x[std::slice(1, N/2, 2)];
 
    // conquer
    fft1(even);
    fft1(odd);
 
    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex_fft t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

double vectorsum(vector<double> v)
{  double  total = 0;
   for (int i = 0; i < v.size(); i++)
      total += v[i];
   return total;
}


// Main
int main(int argc, char** argv)
{
	cout<<"Hello"<<endl;
	std::vector<double> grid1 = linspace(0, 1.0, 6);
	
	cout << "alglib::randomreal() - returns random real number from [0,1)"
              << endl << alglib::randomreal() << endl;
    vector<double> X(5), Y(5);
    X[0]=0.1;
    X[1]=0.4;
    X[2]=1.2;
    X[3]=1.8;
    X[4]=2.0;
    Y[0]=0.1;
    Y[1]=0.7;
    Y[2]=0.6;
    Y[3]=1.1;
    Y[4]=0.9;
    //*** First compute the equispaced X-coordinate points (Following Matlab interprac using armadillo) before using Alglib spline 
	//*** (Need to change the numerical initialization with proper variables)
	//**Performing the Matlab operation diff 
	vector<double> diff_X(5), diff_Y(5);
	adjacent_difference (X.begin(),X.end(),diff_X.begin());
	diff_X.erase(diff_X.begin());
	adjacent_difference (Y.begin(),Y.end(),diff_Y.begin());
	diff_Y.erase(diff_Y.begin());
	vector<double> sqr_diff_X(4), sqr_diff_Y(4), sum_sqr_diff(4),chordlen(4),cumarc(4);
	transform(diff_X.begin(),diff_X.end(),sqr_diff_X.begin(),[](double f)->double { return f * f; });
	transform(diff_Y.begin(),diff_Y.end(),sqr_diff_Y.begin(),[](double f)->double { return f * f; });
	transform(sqr_diff_X.begin(), sqr_diff_X.end(), sqr_diff_Y.begin(),sum_sqr_diff.begin(), plus<double>());
	transform(sum_sqr_diff.begin(), sum_sqr_diff.end(),sum_sqr_diff.begin(), [](double f)->double { return sqrt(f); });
	double sum_of_elems=vectorsum(sum_sqr_diff);
	transform( sum_sqr_diff.begin(), sum_sqr_diff.end(), chordlen.begin(),bind2nd( divides<double>(), sum_of_elems ) );
	partial_sum(chordlen.begin(), chordlen.end(), cumarc.begin());
	cumarc.insert( cumarc.begin(), 0 );
	//***Histc function from armadillo library
	arma::vec Y1(cumarc);
	arma::vec grid2(grid1);
	uvec h= histc(grid2,Y1);
	typedef std::vector<double> stdvec;
	stdvec z = conv_to< stdvec >::from(h); 
	//***Compute index matrix BIN from the Histc output
	int sum_of_z=vectorsum(z);
	vector<int> tbins(sum_of_z);
	int count11=0;
	for ( int j = 0; j < tbins.size()-1; ++j)
	{
		for (int k=0;k<(int)double(z.at(j)); k++)
		{
			tbins.at(count11)=j+1;
			count11=count11+1;
		}
	}
	//***catch any problems at the ends (As per matlab interpac)

	for ( int j = 0; j < tbins.size(); ++j)
	{
		if ((tbins.at(j)<=0)||(grid1.at(j)<=0.0))
		{
			tbins.at(j)=1;
		}
		else if((tbins.at(j)>=5)||(grid1.at(j)>=1.0))
		{
			tbins.at(j)=5-1;
		}
	}
	//*** interpolate(As per matlab interpac)
	vector<double> s(sum_of_z);
	for ( int j = 0; j < s.size(); ++j)
	{
		s.at(j)=(grid1.at(j)-cumarc.at(tbins.at(j)-1))/chordlen.at(tbins.at(j)-1);
	}
	//***Finally generate the equally spaced X-axis coordinated to be used in the alglib spline 
	vector<double> eq_spaced_x(sum_of_z),eq_spaced_y(sum_of_z);
	for ( int j = 0; j < eq_spaced_x.size(); ++j)
	{
		eq_spaced_x.at(j)=X.at(tbins.at(j)-1)+(X.at(tbins.at(j))-X.at(tbins.at(j)-1))*s.at(j);
	}
	//***Alglib spline 
	alglib::real_1d_array AX, AY;
    AX.setcontent(X.size(), &(X[0]));
    AY.setcontent(Y.size(), &(Y[0]));
	alglib::spline1dinterpolant spline;
	//alglib::spline1dbuildcubic(AX, AY, X.size(), 2,0.0,2,0.0, spline);
	alglib::spline1dbuildlinear(AX, AY, X.size(), spline);
	for(size_t i=0; i<X.size(); i++){
      printf("%f %f\n", X[i], Y[i]);
    }
		
	printf("\n");
	double X_try[6]={0.1,0.48,.86,1.24,1.62,2.0};
	//double X_try1[6]={0.1,0.3273,.7429,1.2365,1.6269,2.0};
    for(int i=0; i<6; i++){
      double x=eq_spaced_x[i];
      printf("%f %f\n", x, alglib::spline1dcalc(spline,x));
	  eq_spaced_y.at(i)=alglib::spline1dcalc(spline,x);
    }
    printf("\n");
	//atan2 testing
	vector<double> new_diff_X(eq_spaced_x.size()), new_diff_Y(eq_spaced_y.size());
	adjacent_difference (eq_spaced_x.begin(),eq_spaced_x.end(),new_diff_X.begin());
	new_diff_X.erase(new_diff_X.begin());
	adjacent_difference (eq_spaced_y.begin(),eq_spaced_y.end(),new_diff_Y.begin());
	new_diff_Y.erase(new_diff_Y.begin());
	std::valarray<double> ycoords (new_diff_Y.data(),new_diff_Y.size());
    std::valarray<double> xcoords (new_diff_X.data(),new_diff_X.size());
	std::valarray<double> results = atan2 (xcoords,ycoords);
	vector<double> Curvature(eq_spaced_x.size()-1);
	Curvature.assign(std::begin(results), std::end(results));
	double tempsub=Curvature.at(0);
	for(int i=0; i<Curvature.size()-1; i++){
		
		double temp_sum=vectorsum(sum_sqr_diff);
		Curvature.at(i)=Curvature.at(i)-tempsub;
	}
	//double* fftarr = &Curvature[0];
	
	//CArray_fft fft_data(Curvature.at(0), Curvature.size());
	int num=4;
	Complex_fft* fftarr1=new Complex_fft[num];
	for(int i=0; i<Curvature.size()-1; i++){//to test with 4 i.e. power of 2 number of elements
		fftarr1[i]=std::complex<double>(Curvature.at(i),0.0);
		cout<<fftarr1[i]<<endl;
	}
	CArray_fft fftarr (fftarr1,4);

	

	//const Complex_fft test[] = { 0.0,0.8651,1.1612,0.4124 };//,1.2554
	//CArray_fft data(test, 4);
	fft1(fftarr);
    vector<double> fX(num);
    std::cout << "fft" << std::endl;
    for (int i = 0; i < num; ++i)//put num
    {
        std::cout << fftarr[i] << std::endl;
		std::complex<double> temp_comp;
		temp_comp=(fftarr[i])*conj(fftarr[i]);
		fX.at(i)=temp_comp.real();
    }
	vector<double> Fet_descrp(num-1);//*make sure here 'num' is even number
	for (int i = 0; i < num-1; ++i)
	{
		if (fX.at(i+1)>fX.at(i))
		{
			vector<double> tempintrvl=linspace_intrvl(fX.at(i),fX.at(i+1),1);			
			double temp_sum=vectorsum(tempintrvl);
			Fet_descrp.at(i)=temp_sum;
		}
		else
		{
			Fet_descrp.at(i)=0.0;
		}
	}

	//***Polybool of matlab testing using ploygon intersection & Union operation
	// Define a polygons and fill the outer rings.
	double points[][2] = {{2.0, 1.3}, {4.1, 3.0}, {5.3, 2.6}, {2.9, 0.7}, {2.0, 1.3}};
	model::polygon<model::d2::point_xy<double> > poly;
	append(poly, points);
	boost::tuple<double, double> p = boost::make_tuple(3.7, 2.0);
	boost::tuple<double, double> p1 = boost::make_tuple(0.0, 0.0);
	std::cout << "Point p is in polygon? " << std::boolalpha << within(p, poly) << std::endl;
	std::cout << "Point p1 is in polygon? " << std::boolalpha << within(p1, poly) << std::endl;
    polygon_2d a;
    {
        const double c[][2] = {
            {160, 330}, {60, 260}, {20, 150}, {60, 40}, {190, 20}, {270, 130}, {260, 250}, {160, 330}
        };
        append(a, c);
    }
    correct(a);
    cout << "A: " << dsv(a) << std::endl;	
	cout << "Area of A " << boost::geometry::area(a) << std::endl;
	polygon_2d b;
    {
        const double c[][3] = {
            {300, 330}, {190, 270}, {150, 170}, {150, 110}, {250, 30}, {380, 50}, {380, 250}, {300, 330}
        };
        append(b, c);
    }
    correct(b);
    std::cout << "B: " << dsv(b) << std::endl;
	cout << "Area of B " << boost::geometry::area(b) << std::endl;
	// Calculate interesection
    std::deque<polygon> output;
	//polygon_2d output;
	//std::vector<Polygon> output;
    boost::geometry::intersection(a, b, output);
	int iintr = 0;
    std::cout << "Intersection area:" << std::endl;
	std::vector< double > X_intrsc_Points;
	std::vector< double > Y_intrsc_Points;
	
    BOOST_FOREACH(polygon const& p1, output)
    {
        std::cout << iintr++ << ": " << boost::geometry::area(p1) << " with Points" << dsv(p1)<< std::endl;
		for (auto it1 = boost::begin(boost::geometry::exterior_ring(p1)); it1 != boost::end(boost::geometry::exterior_ring(p1)); ++it1)
			{
				//cout<<get<0>(*it1)<<endl;
				X_intrsc_Points.push_back(get<0>(*it1));
				//cout<<get<1>(*it1)<<endl;
				Y_intrsc_Points.push_back(get<1>(*it1));
			}
    }

	// Calculate Union
    std::deque<polygon> output1;
    boost::geometry::union_(a, b, output1);
	int iuni = 0;
    std::cout << "Union area:" << std::endl;
	std::vector< double > X_unin_Points;
	std::vector< double > Y_unin_Points;
    BOOST_FOREACH(polygon const& p2, output1)
    {
        std::cout << iuni++ << ": " << boost::geometry::area(p2)<< " with Points" << dsv(p2) << std::endl;
		for (auto it2 = boost::begin(boost::geometry::exterior_ring(p2)); it2 != boost::end(boost::geometry::exterior_ring(p2)); ++it2)
			{
				//cout<<get<0>(*it1)<<endl;
				X_unin_Points.push_back(get<0>(*it2));
				//cout<<get<1>(*it1)<<endl;
				Y_unin_Points.push_back(get<1>(*it2));
			}

    }






	//*** bintprog of matlab testing using Cbc C++ Mixed integer programming library	
	//example from link (projects.coin-or.org/Cbc/wiki/VSSetup)
	/*
	const int numcols = 2;
	const int numrows = 1;
	double obj[] = { 1.0, 1.0}; // obj: Max x0 + x1
  
	// Column-major sparse "A" matrix: x0 + 2 x1 <= 3.9
	int start[] = {0, 1, 2};      // where in index columns start (?)
	int index[] = {0, 0};         // row indexs for the columns
	double values[] = {1.0, 2.0}; // the values in the sparse matrix
	double rowlb[]  = {0.0};
	double rowub[]  = {3.9};

	//          0 <= x0 <= 10 and integer
	//          0 <= x1 <= 10
	double collb[] = {0.0, 0.0};
	double colub[] = {10.0, 10.0};

	//
	OsiClpSolverInterface optmodel;
    optmodel.loadProblem(numcols, numrows, start, index, values, 
                    collb, colub, obj, rowlb, rowub);
	optmodel.setInteger(0); // Sets x0 to integer
	//optmodel.setInteger(1); optmodel.setInteger(2); optmodel.setInteger(3);// Sets x1 to integer
    optmodel.setObjSense(-1.0); //-1.0 Maximise, 1.0 for minimize

	CbcModel solver(optmodel);
	solver.branchAndBound();
	bool optimal = solver.isProvenOptimal();
	const double *val = solver.getColSolution();
	printf("Solution %g %g\n", val[0], val[1]);

	*/
	//Equivalent Matlab bintprog
	/*
	f1 = [-5; -4; -3];
    A1 = [2 3 1; 4 1 2;3 4 2];
    b1 = [5; 11; 8];
    [X,FVAL,EXITFLAG,OUTPUT] = bintprog(f1,A1,b1)
	or
	f = [-9; -5; -6; -4];
    A = [6 3 5 2; 0 0 1 1; -1 0 1 0; 0 -1 0 1];
    b = [9; 1; 0; 0];
	For C++ version:
	**OsiClpSolverInterface
	**setInteger
	**branchAndBound
	*/
	//
	OsiClpSolverInterface* si = new OsiClpSolverInterface();
    //Number of variables (columns) in problem is three.
    //int n_cols = 3;
	int n_cols = 4;
    double * objective    = new double[n_cols];//the objective coefficients
    double * col_lb       = new double[n_cols];//the column lower bounds
    double * col_ub       = new double[n_cols];//the column upper bounds

    //Define the objective coefficients.
    //minimize 5 x0 + 4 x1 + 3 x2
    //objective[0] = 5.0;
    //objective[1] = 4.0;
    //objective[2] = 3.0;
	objective[0] = 9.0; objective[1] = 5.0; objective[2] = 6.0; objective[3] = 4.0;

    //Define the variable lower/upper bounds.
    // x0 >= 0   =>  0 <= x0 <= infinity
    // x1 >= 0   =>  0 <= x1 <= infinity
    // x2 >= 0   =>  0 <= x2 <= infinity
    col_lb[0] = 0.0;
    col_lb[1] = 0.0;
    col_lb[2] = 0.0;
	col_lb[3] = 0.0;
    col_ub[0] = 1.0;//si->getInfinity();//Converting to integer programming
    col_ub[1] = 1.0;//si->getInfinity();
    col_ub[2] = 1.0;//si->getInfinity();
	col_ub[3] = 1.0;
     
    //Number of inequalities (rows) in problem is three.
    //int n_rows = 3;
	int n_rows = 4;
    double * row_lb = new double[n_rows]; //the row lower bounds
    double * row_ub = new double[n_rows]; //the row upper bounds
     
    //Define the constraint matrix.
    CoinPackedMatrix * matrix =  new CoinPackedMatrix(false,0,0);
    matrix->setDimensions(0, n_cols);

    //2 x0 + 3 x1 + 1 x2 <= 5  =>  -infinity <= 2 x0 + 3 x1 + 1 x2 <= 5
    CoinPackedVector row1;
    //row1.insert(0, 2.0);//(index,coeff)
    //row1.insert(1, 3.0);
    //row1.insert(2, 1.0);
	row1.insert(0, 6.0); row1.insert(1, 3.0); row1.insert(2, 5.0); row1.insert(3, 2.0);
    row_lb[0] = -1.0 * si->getInfinity();
    //row_ub[0] = 5.0;
	row_ub[0] = 9.0;
    matrix->appendRow(row1);

    //4 x0 + 1 x1 + 2 x2 <= 11  =>  -infinity <= 4 x0 + 1 x1 + 2 x2 <= 11
    CoinPackedVector row2;
    //row2.insert(0, 4.0);
    //row2.insert(1, 1.0);
    //row2.insert(2, 2.0);
	row2.insert(0, 0.0); row2.insert(1, 0.0); row2.insert(2, 1.0); row2.insert(3, 1.0);
    row_lb[1] = -1.0 * si->getInfinity();
    //row_ub[1] = 11.0;
	row_ub[1] = 1.0;
    matrix->appendRow(row2);

    //3 x0 + 4 x1 + 2 x2 <= 8  =>  -infinity <= 3 x0 + 4 x1 + 2 x2 <= 8
    CoinPackedVector row3;
    //row3.insert(0, 3.0);
    //row3.insert(1, 4.0);
    //row3.insert(2, 2.0);
	row3.insert(0, -1.0); row3.insert(1, 0.0); row3.insert(2, 1.0); row3.insert(3, 0.0);
    row_lb[2] = -1.0 * si->getInfinity();
    //row_ub[2] = 8.0;
	row_ub[2] = 0.0;
    matrix->appendRow(row3);
    
	//4th Constraint
    CoinPackedVector row4;
    //row3.insert(0, 3.0);
    //row3.insert(1, 4.0);
    //row3.insert(2, 2.0);
	row4.insert(0, 0.0); row4.insert(1, -1.0); row4.insert(2, 0.0); row4.insert(3, 1.0);
    row_lb[3] = -1.0 * si->getInfinity();
    //row_ub[2] = 8.0;
	row_ub[3] = 0.0;
    matrix->appendRow(row4);

    //load the problem to OSI
    si->loadProblem(*matrix, col_lb, col_ub, objective, row_lb, row_ub);

    //write the MPS file to a file called example.mps
    si->writeMps("example");
    
    
    //we want to maximize the objective function
    si->setObjSense(-1);
	si->setInteger(0); si->setInteger(1); si->setInteger(2);
	si->setInteger(3);
    //solve the linear program
    si->branchAndBound();//initialSolve();

    const double * solution = si->getColSolution();
    
    //get, print the solution
    const double objective_value = si->getObjValue();

    //Don't mix the index range used here (x0, x1, x2) with that
    //used in Chvatal's book (x1, x2, x3)!
    cout << "\nThe optimal solution is:" << endl
    << "  x0 = " << solution[0] << endl
    << "  x1 = " << solution[1] << endl
    << "  x2 = " << solution[2] << endl
	<< "  x2 = " << solution[3] << endl
    << "  objective value = " << objective_value << endl;

    OsiClpSolverInterface* si2 = new OsiClpSolverInterface();
    si2->readMps("example");
    si2->initialSolve();
    

    //free the memory
    if(objective != 0)   { delete [] objective; objective = 0; }
    if(col_lb != 0)      { delete [] col_lb; col_lb = 0; }
    if(col_ub != 0)      { delete [] col_ub; col_ub = 0; }
    if(row_lb != 0)      { delete [] row_lb; row_lb = 0; }
    if(row_ub != 0)      { delete [] row_ub; row_ub = 0; }
    if(matrix != 0)      { delete matrix; matrix = 0; }

	system("pause");
	return 0;
}