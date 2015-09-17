#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "baseFunc.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <afxwin.h>
#include <iostream>
#include <vector>
#include <mat.h>
#include <algorithm>
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
#include <numeric> 
#include <iterator>
#include <armadillo>
#include <cstddef>      // std::size_t
#include <cmath>        // std::atan2
#include <limits>
#include <valarray> 
#include <complex>
#include <functional>
#include "dirent.h"
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

#ifndef isnan
inline bool isnan(double x) {
    return x != x;
}
#endif

#include "CoinPackedVector.hpp"
#include "CoinPackedMatrix.hpp"

BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)
const double PI = 3.141592653589793238460;


typedef std::complex<double> Complex_fft;
typedef std::valarray<Complex_fft> CArray_fft;

bool IsNonZero (int i) { return ((i!=0)==1); }

using namespace std;
using namespace cv;
using namespace flann;
using namespace alglib;
using namespace arma;
using namespace alglib_impl;
using namespace boost::geometry;
using boost::geometry::get;

namespace bg = boost::geometry;
typedef bg::model::d2::point_xy<double> point_2d;
typedef bg::model::polygon<point_2d> polygon_2d;
typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;

typedef std::pair<double,int> mypair;
static bool sort_using_greater_than(mypair u, mypair v)
{
   return u.first > v.first;
}


std::vector<std::vector<double> > vec_transpose(const std::vector<std::vector<double> > data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    std::vector<std::vector<double> > result(data[0].size(),
                                          std::vector<double>(data.size()));
    for (std::vector<double>::size_type i = 0; i < data[0].size(); i++) 
        for (std::vector<double>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;

}



std::vector<std::vector<int> > vec_transpose_4_int(const std::vector<std::vector<int> > data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    std::vector<std::vector<int> > result(data[0].size(),
                                          std::vector<int>(data.size()));
    for (std::vector<int>::size_type i = 0; i < data[0].size(); i++) 
        for (std::vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;

}


std::vector<double> linspace_intrvl(double first, double last, double intrvl) {
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

std::vector<int> nonzeroitems_vec(vector<int> v)
{  
   vector<int> result;
   for (int i = 0; i < v.size(); i++)
   {
	   if (v[i]!=0)
	   {
		   result.push_back(v[i]);
	   }
   } 
   return result;
}

bool CheckCommon(std::vector<int> const& inVectorA, std::vector<int> const& nVectorB)
{
    return std::find_first_of (inVectorA.begin(), inVectorA.end(),
                               nVectorB.begin(), nVectorB.end()) != inVectorA.end();
}


double vectorsqrsum(vector<double> v)
{  double  total = 0;
   for (int i = 0; i < v.size(); i++)
      total += v[i]*v[i];
   return total;
}

void normalizevector(vector<double> v,vector<double> &v1)
{
	auto biggest = std::max_element(std::begin(v), std::end(v));	
	double tempbig=*biggest;
	for (int i = 0; i < v.size(); i++)
	{
      v1.push_back(v[i]/tempbig);
	}

}
void boundary_disp(vector<Point> v, Point a,vector<double> &v1,vector<double> &v2)
{  
   for (int i = 0; i < v.size(); i++)
   {
	   
	   v1.push_back(v[i].x+a.y);
	   v2.push_back(v[i].y+a.x);
	   
   }
      
}

double medianofvector(vector<double> scores)
{
  double median;
  size_t size = scores.size();

  sort(scores.begin(), scores.end());

  if (size  % 2 == 0)
  {
      median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  }
  else 
  {
      median = scores[size / 2];
  }

  return median;
}

std::vector<double> linspace(double first, double last, int len) {
  std::vector<double> result(len);
  double step = (last-first) / (len - 1);
  for (int i=0; i<len; i++) { result[i] = first + i*step; }
  return result;
}

void matread(const char *file, std::vector<double>& v)
{
    // open MAT-file
    MATFile *pmat = matOpen(file, "r");
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray *arr = matGetVariable(pmat, "bwVesselMask" );//"name of the variable to be read in the *.mat file"
    if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
        // copy data
        mwSize num = mxGetNumberOfElements(arr);
        double *pr = mxGetPr(arr);
        if (pr != NULL) {
            v.resize(num);
            v.assign(pr, pr+num);
        }
    }

    // cleanup
    mxDestroyArray(arr);
    matClose(pmat);
}

void read_and_dialate(const char *imagename,const char *seg_file1,cv::Mat &img341)
	{
		cv::Mat imgGray;
		cv::Mat img = imread(imagename);
		Size s = img.size();
		cout<<"Image Height:"<<s.height<<endl;
		cout<<"Image Width:"<<s.width<<endl;
		Size size(s.height/2,s.width/2);

		//Read the corresponding segmentation .mat file
	    std::vector<double> v;
        matread(seg_file1, v);
		cv::Mat mymat=cv::Mat(v);
		//resizing the mat
	    cv::Mat mat_dst(s.width,s.height,CV_64FC1);//Read the image size
	    int k=0;
	    for(int i=0; i<s.width; ++i)
		{
			for(int j=0; j<s.height; ++j)
			{
				mat_dst.at<double>(i,j)=mymat.at<double>(k);
				k=k+1;
			}
		}	 
		cout<<"ReSize::"<<mat_dst.size()<<endl;
		//Transpose the mat
		mat_dst=mat_dst.t();
		cout<<"Transpose::"<<mat_dst.size()<<endl;
		//Finding location of 1
		int non_zeros=sum(mat_dst)[0];
		int *row=new int[non_zeros];
		int *col=new int[non_zeros];
		int count=0;
		for (int i = 0; i < s.height; i++ ) {
			for (int j = 0; j < s.width; j++) {
				if (mat_dst.at<double>(i, j) == 1) { 
					row[count]=i;
					col[count]=j;
					count=count+1;
            //cout << i << ", " << j << endl;     // Do your operations
					}
			}
		}
	    cv::dilate(mat_dst, img341, getStructuringElement(MORPH_RECT, Size (5,5)));//Perform Image dialation on the RGB image			

	}

void get_boundary_centroid(cv::Mat dialated,vector<vector<Point>> &blobs_sorted, vector<double> &bolb_area_sorted,vector<Point> &bolb_center1_sorted, vector<Vec4i> &hierarchy)
{

			int top=10;
			//comment if not returning the sorted values
			vector<vector<Point>> blobs;
			vector<double> bolb_area;
			vector<Point> bolb_center1;

			//Uncomment if not returning the sorted values
			//vector<Point> bolb_center1_sorted;
			//vector<vector<Point>> blobs_sorted;

			cv::Mat img225;
			dialated.convertTo(img225,CV_8U);
			findContours(img225,blobs,hierarchy,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);//*** Find contour matches with Matlab better then labelBolbs
			//**Find the center of the bolbs & the areas

	
			for (int i = 0; i < blobs.size(); i++ ) {
			float sum_x=0;
			float sum_y=0;
			for (int j = 0; j < blobs[i].size(); j++) {
				sum_x=sum_x+blobs[i][j].x;
				sum_y=sum_y+blobs[i][j].y;			
			}
			
			bolb_area.push_back(contourArea(blobs[i]));
			//**bolb center by Moments
			bolb_center1.push_back(Point(int(moments(blobs[i]).m10/moments(blobs[i]).m00),int(moments(blobs[i]).m01/moments(blobs[i]).m00)));
		
			}
			// This is a vector of {value,index} pairs
			vector<pair<double,size_t> > vp;
			vp.reserve(bolb_area.size());
			for (size_t i = 0 ; i != bolb_area.size() ; i++) {
				vp.push_back(make_pair(bolb_area[i], i));
			}
			// Sorting will put lower values ahead of larger ones,
			// resolving ties using the original index
			sort(vp.begin(), vp.end(),sort_using_greater_than);

			//Xor operation in matlab
			/*
			vector<int> tmp_inx;
			for (int i = 0; i < top; i++ ) {
				tmp_inx.push_back(vp[i].second);
			}			
			sort(tmp_inx.begin(), tmp_inx.end());
			*/
			// Until-here

			for (int i = 0; i < bolb_area.size(); i++ ) {//** Guess Not top 30 but choose the points whose area greater then 30//
				bolb_center1_sorted.push_back(bolb_center1.at(vp[i].second));//*** Better match with Matlab version
				//bolb_center1_sorted.push_back(bolb_center1.at(tmp_inx[i]));
				
				}
			
			for (int i = 0; i < top; i++ ) {//** Guess Not top 30 but choose the points whose area greater then 30 //bolb_area.size()
				blobs_sorted.push_back(blobs.at(vp[i].second));//*** Better match with Matlab version
				bolb_area_sorted.push_back(bolb_area.at(vp[i].second));
				//blobs_sorted.push_back(blobs.at(tmp_inx[i]));
				//bolb_area_sorted.push_back(bolb_area.at(tmp_inx[i]));

			}

}

void get_FSDs(vector<double> X,vector<double> Y, vector<double> &Fsd)
{
	int num_intrp=65;
	std::vector<double> grid1 = linspace(0, 1.0, num_intrp);//Equivalent to (0:1/n-1:1) in matlab
	//**Performing the Matlab operation diff 
	vector<double> diff_X(X.size()), diff_Y(Y.size());
	adjacent_difference (X.begin(),X.end(),diff_X.begin());
	diff_X.erase(diff_X.begin());
	adjacent_difference (Y.begin(),Y.end(),diff_Y.begin());
	diff_Y.erase(diff_Y.begin());
	vector<double> sqr_diff_X(X.size()-1), sqr_diff_Y(X.size()-1), sum_sqr_diff(X.size()-1),chordlen((X.size())-1),cumarc((X.size())-1);
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
	for (int k=0;k<z.size(); k++)
	{
		if (z.at(k)!=0)
		{
			tbins.at(count11)=k;
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
		else if((tbins.at(j)>=X.size())||(grid1.at(j)>=1.0))
		{
			tbins.at(j)=X.size()-1;
		}
	}
	//*** interpolate(As per matlab interpac)
	vector<double> s1(sum_of_z);
	for ( int j = 0; j < s1.size(); ++j)
	{
		s1.at(j)=(grid1.at(j)-cumarc.at(tbins.at(j)-1))/chordlen.at(tbins.at(j)-1);
	}
	//***Finally generate the equally spaced X-axis coordinated to be used in the alglib spline 
	vector<double> eq_spaced_x(sum_of_z),eq_spaced_y(sum_of_z);
	vector<Point> blobs_intrp;//Later need to change to data-type "vector<vector<Point>>" to hold for all other vessel objects
	for ( int j = 0; j < eq_spaced_x.size(); ++j)
	{
		eq_spaced_x.at(j)=X.at(tbins.at(j)-1)+(X.at(tbins.at(j))-X.at(tbins.at(j)-1))*s1.at(j);
		eq_spaced_y.at(j)=Y.at(tbins.at(j)-1)+(Y.at(tbins.at(j))-Y.at(tbins.at(j)-1))*s1.at(j);
		blobs_intrp.push_back(Point(eq_spaced_x.at(j),eq_spaced_y.at(j)));
		//circle( img226,  cv::Point(eq_spaced_x.at(j),eq_spaced_y.at(j)), 1.0, Scalar( 0, 0, 255 ), 1, 8 );
	}
	//FSD computation
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

	Complex_fft* fftarr1=new Complex_fft[num_intrp-1];
	for(int i=0; i<Curvature.size()-1; i++){//to test with 4 i.e. power of 2 number of elements
		fftarr1[i]=std::complex<double>(Curvature.at(i),0.0);
	}
	CArray_fft fftarr (fftarr1,num_intrp-1);
	fft1(fftarr);
	vector<double> fX(num_intrp-1);
	for (int i = 0; i < num_intrp-1; ++i)//put num
    {
		std::complex<double> temp_comp;
		temp_comp=(fftarr[i])*conj(fftarr[i]);
		fX.at(i)=temp_comp.real();
    }
	vector<double> obj_fsd(num_intrp-2);//*make sure here 'num' is even number
	for (int i = 0; i < num_intrp-2; ++i)
	{
		if (fX.at(i+1)>fX.at(i))
		{
			vector<double> tempintrvl=linspace_intrvl(fX.at(i),fX.at(i+1),1);			
			double temp_sum=vectorsum(tempintrvl);
			obj_fsd.at(i)=temp_sum;
		}
		else
		{
			obj_fsd.at(i)=0.0;
		}
	}
	Fsd=obj_fsd;
}
void get_likelihood_value(vector<double> feature1,Point centroid1,double area1,vector<double> feature2,Point centroid2,double area2,vector<double> cur_vec,vector<double> pre_vec_j,int type, int problem_id, double &like_values)
{	
	//define 'k' if needed
	int k=round(feature1.size()/2);
	k=k+1;
	double delta1 = 50000;
	double delta2 = 50;
	double delta3 = 500;
	double omega3 = 0;
	double omega1;
	double omega2;
	double  cos_theta1;
	double  cos_theta2;
	//feature difference
	vector<double> diff_feature_temp;
	vector<double> feature_temp1=feature1;
	vector<double> feature_temp2=feature2;
	feature_temp1.resize(k);
	feature_temp2.resize(k);
	std::transform(feature_temp1.begin(), feature_temp1.end(), feature_temp2.begin(), std::back_inserter(diff_feature_temp),
	[](double feature1, double feature2) { return fabs(feature1-feature2); });
	double diff_feature_sqr=vectorsqrsum(diff_feature_temp);
	double diff_feature=sqrt(diff_feature_sqr);
	//distance between centroids
	double diff_centroid_square=((centroid1.x-centroid2.x)*(centroid1.x-centroid2.x)+(centroid1.y-centroid2.y)*(centroid1.y-centroid2.y));
	double diff_centroid=sqrt(diff_centroid_square);
	// area difference
	double diff_area=fabs(area1-area2);
	if(problem_id==2)
	{
		Point vec=centroid2-centroid1;
		//Code the rest ...
		if((pre_vec_j[0]==1)&&(pre_vec_j[1]==1))
		{
			cos_theta1 = 0;
		}
		else
		{
			cos_theta1=((vec.x*pre_vec_j[0])+(vec.y*pre_vec_j[1]))/((sqrt(double(vec.x)*double(vec.x)+double(vec.y)*double(vec.y)))*(sqrt((pre_vec_j[0]*pre_vec_j[0])+(pre_vec_j[1]*pre_vec_j[1]))));
		}
		if ((isnan(cur_vec[0]))&&(isnan(cur_vec[1])))
		{
			cos_theta2 = 0;
		}
		else
		{
			cos_theta2=((cur_vec[0]*pre_vec_j[0])+(cur_vec[1]*pre_vec_j[1]))/((sqrt(cur_vec[0]*cur_vec[0]+cur_vec[1]*cur_vec[1]))*(sqrt((pre_vec_j[0]*pre_vec_j[0])+(pre_vec_j[1]*pre_vec_j[1]))));
		}
		if((cos_theta1!=0)&&(cos_theta2!=0))
		{
			omega3 = 0.2;
		}
	}
	//***Plug in code for problemid==2 check
	if(omega3!=0)
	{
		//***Plugin the code
		//For type=1(stands for 1to1)
		if(type==1)
		{
			omega1 = 8 * omega3;
			omega2 = 4 * omega3;
		}
		//For type!=1(stands for 1to2)
		else
		{
			omega1 = 6 * omega3;
			omega2 = 4 * omega3;
		}
		
	}
	else
	{
		//For type=1
		if(type==1)
		{
			omega1 = 5;
			omega2 = 3;
		}
		//For type!=1(stands for 1to2)
		else
		{
			omega1 = 6;
			omega2 = 4;
		}
		
		//***Plugin the code for type!=1
	}

	double omegas[3]={omega1/(omega1+omega2+omega3),omega2/(omega1+omega2+omega3),omega3/(omega1+omega2+omega3)};
					
	//***Plug in code for problemid==2 check
	if(problem_id==2)
	{
		//Check For type=1
		if(type==1)
		{
			like_values=omegas[0]*exp(-diff_feature/delta1)+omegas[1]*exp(-diff_centroid/delta2)+omegas[2]*exp(acos(cos_theta1)-acos(cos_theta2));
		}

		//Check For type=2
		else
		{
			like_values=omegas[0]*exp(-diff_feature/delta1)+omegas[1]*exp(-diff_centroid/delta2)+omegas[2]*exp(acos(cos_theta1)-acos(cos_theta2));
		}
	}
	else
	{
		//Check For type=1
		if(type==1)
		{
			like_values=omegas[0]*exp(-diff_feature/delta1)+omegas[1]*exp(-diff_centroid/delta2);
		}

		//Check For type=2
		else
		{
			like_values=omegas[1]*exp(-diff_feature/delta1)+omegas[2]*exp(-diff_centroid/delta2);
		}

	}		

}

void calculate_likelihood(int i,vector<vector <double>> Fsd_slide_1,vector<Point> bolbcenter_1, vector <double> bolbarea_1,vector<vector <double>> Fsd_slide_2,vector<Point> bolbcenter_2, vector <double> bolbarea_2,int problem_id,vector<vector<Point>> blobs_2,vector<double> pre_vec_j,vector<vector<double>> vec3,vector<double> &norm_p_likelihood, vector<vector<int>> &C_coeff)//a 'vec3' object for problemid=2
{
	vector<double> p_likelihood;
	int N1=bolbcenter_1.size();
	int N2=bolbcenter_2.size();
	vector<double> feature1 = Fsd_slide_1.at(i);
	Point centroid1 = bolbcenter_1.at(i);
	double area1 =bolbarea_1.at(i);
	//One-to-One
	for (int j = 0; j < blobs_2.size(); j++ ){
		vector<double> feature2 = Fsd_slide_2.at(j);
		Point centroid2 = bolbcenter_2.at(j);
		double area2 =bolbarea_2.at(j);
		vector<double> cur_vec;
		if (problem_id==2)
		{
			cur_vec =vec3[j];
		}
		double like_values;
		get_likelihood_value(feature1,centroid1,area1,feature2,centroid2,area2,cur_vec,pre_vec_j,1, problem_id, like_values);  //int type, int problem_id (last not two)
		if(like_values!=0)
		{
			p_likelihood.push_back(like_values);
			std::vector<int> c_temp(N1+N2, 0);
			c_temp[i]=1;
			c_temp[N1+j]=1;
			C_coeff.push_back(c_temp);
		}
	}
	//One-to-two
	//K & Intervals inside the get_FSDs function
	for (int j = 0; j < blobs_2.size(); j++ ){
		int obj1_id=j;
		Point centroid_1 = bolbcenter_2.at(j);
		vector<Point> boundary1=blobs_2.at(j);

		for (int k = j+1; k < blobs_2.size(); k++ ){
						
			Point centroid_2 = bolbcenter_2.at(k);
			Point avg_centroid;
			avg_centroid.x=(centroid_1.x+centroid_2.x)/2;
			avg_centroid.y=(centroid_1.y+centroid_2.y)/2;
			Point dis_1=avg_centroid-centroid_1; //displacement of the 1st object

			int obj2_id=k;
			vector<Point> boundary2 = blobs_2.at(obj2_id);
			Point dis_2=avg_centroid-centroid_2; //displacement of the 2nd object

			// if the two objects intersects after being moved to the average
			vector<double> disp_boundary1_x;
			vector<double> disp_boundary1_y;
			vector<double> disp_boundary2_x;
			vector<double> disp_boundary2_y;
			boundary_disp( boundary1, dis_1,disp_boundary1_x,disp_boundary1_y);
			boundary_disp( boundary2, dis_2,disp_boundary2_x,disp_boundary2_y);
						
			polygon_2d bound1;
			{
				const double c[][2] = {
					{160, 330}, {60, 260}, {20, 150}, {60, 40}, {190, 20}, {270, 130}, {260, 250}, {160, 330}
				};
							
				//append(bound1, c);
							
				vector<double>::const_iterator xi;
				vector<double>::const_iterator yi;
				for (xi=disp_boundary1_x.begin(), yi=disp_boundary1_y.begin(); xi!=disp_boundary1_x.end(); ++xi, ++yi)
				{
					append(bound1, make<point_2d>(*xi, *yi));
				}
							
			}
			correct(bound1);
			//cout << "bound1: " << dsv(bound1) << std::endl;
			//cout << "Area of bound1 " << boost::geometry::area(bound1) << std::endl;
			polygon_2d bound2;
			{

				const double c[][3] = {
				{300, 330}, {190, 270}, {150, 170}, {150, 110}, {250, 30}, {380, 50}, {380, 250}, {300, 330}
				};
				//append(bound2, c);
							
				vector<double>::const_iterator xi;
				vector<double>::const_iterator yi;
				for (xi=disp_boundary2_x.begin(), yi=disp_boundary2_y.begin(); xi!=disp_boundary2_x.end(); ++xi, ++yi)
				{
					append(bound2, make<point_2d>(*xi, *yi));
				}
							
			}
			correct(bound2);
			//cout << "bound2: " << dsv(bound2) << std::endl;
			//cout << "Area of bound2 " << boost::geometry::area(bound2) << std::endl;
			// Calculate interesection
			std::deque<polygon> intersect;
			try
			{
				boost::geometry::intersection(bound1, bound2, intersect);
			}catch(const boost::geometry::overlay_invalid_input_exception){
					cout<<intersects(bound1)<<endl;
					cout<<intersects(bound2)<<endl;
					break;
				}
			double intrsect_area=0;
			int iintr = 0;
			BOOST_FOREACH(polygon const& p1, intersect)
			{
				intrsect_area=boost::geometry::area(p1);
				//std::cout << iintr++ << ": " << boost::geometry::area(p1) << " with Points" << dsv(p1)<< std::endl;
			}

			if (intrsect_area!=0)
			{
				// Calculate Union
				std::deque<polygon> unionpoly;
				try
				{
					boost::geometry::union_(bound1, bound2, unionpoly);
				} catch(const boost::geometry::overlay_invalid_input_exception){
					cout<<intrsect_area<<endl;
					cout<<intersects(bound1)<<endl;
					cout<<intersects(bound2)<<endl;
					break;
				}


				
				int iuni = 0;
				//std::cout << "Union area:" << std::endl;
				std::vector< double > com_boundary_x;
				std::vector< double > com_boundary_y;
				
				double poly_area=0;
				BOOST_FOREACH(polygon const& p2, unionpoly)
				{
					poly_area=boost::geometry::area(p2);
					//std::cout << iuni++ << ": " << boost::geometry::area(p2)<< " with Points" << dsv(p2) << std::endl;
					
					
					for (auto it2 = boost::begin(boost::geometry::exterior_ring(p2)); it2 != boost::end(boost::geometry::exterior_ring(p2)); ++it2)
						{
							//cout<<get<0>(*it1)<<endl;
							com_boundary_x.push_back(get<0>(*it2));
							//cout<<get<1>(*it1)<<endl;
							com_boundary_y.push_back(get<1>(*it2));
						}
				}

				std::vector< double > test_orientation;
				for (int ior = 0; ior < com_boundary_x.size()-1; ior++)
				   {
						
					   test_orientation.push_back((com_boundary_x[ior+1]-com_boundary_x[ior])+(com_boundary_y[ior+1]+com_boundary_y[ior]));
	   
				   }
				if (vectorsum(test_orientation)>0)
				{
					//cout<<"Union Resulted in clockwise polygon"<<endl;
				}
				else
				{
					cout<<"*** Union Resulted in Counter-clockwise polygon***"<<endl;
				}

				vector<double> comFsd;
				get_FSDs(com_boundary_x,com_boundary_y, comFsd);
				vector<double> cur_vec;
				if (problem_id == 2)
				{
					cur_vec =vec3[k];
				}
				double like_values1;
				get_likelihood_value(feature1,centroid1,area1,comFsd,avg_centroid,poly_area,cur_vec,pre_vec_j,2, problem_id, like_values1);  //int type, int problem_id (last not two)
				
				if(like_values1!=0)
				{
					p_likelihood.push_back(like_values1);
					std::vector<int> c_temp(N1+N2, 0);
					c_temp[i]=1;
					c_temp[N1+j]=1;
					c_temp[N1+k]=1;
					C_coeff.push_back(c_temp);
				}
				

			}
		}
				
	}
	//One-to-none
	double like_values2=medianofvector(p_likelihood);	
	
	if(like_values2!=0)
	{
		p_likelihood.push_back(like_values2);
		std::vector<int> c_temp(N1+N2, 0);
		c_temp[i]=1;
		C_coeff.push_back(c_temp);
	}
	

	//normalization
	/*
	vector<double> normvec;
	static const double v_[] = {1,2,3,4,5};
	std::vector<double> vec11 (v_, v_ + sizeof(v_) / sizeof(v_[0]) );
	normalizevector(vec11,normvec);
	*/
	//vector<double> norm_p_likelihood;
	normalizevector(p_likelihood,norm_p_likelihood);

}

void get_matching_result(vector<double> vsol,vector<vector<int>> C_coeff,vector<vector <double>> Fsd_slide_1,vector<vector <double>> Fsd_slide_2,vector<Point> bolbcenter_1,vector<Point> bolbcenter_2, vector<int> &parent,vector<vector<int>> &t_id)
{
	//find selected hypothesis
	//cout<<parent.size()<<endl;
	vector<vector<int>> hypo;
	for (int i = 0; i < vsol.size(); i++ ) 
	{
		if (vsol[i]>0.5)
		{
			hypo.push_back(C_coeff.at(i));
		}
	}
	int n1=Fsd_slide_1.size();
	int n2=Fsd_slide_2.size();
	int N1=bolbcenter_1.size();
	int N2=bolbcenter_2.size();
						
	//Run a for loop of size hypo 
	for (int i = 0; i < hypo.size(); i++ )
	{
		vector <int> temp_t_id;
		vector <int> temp_indx;
		for (int j = 0; j < hypo[i].size(); j++ ) 
		{
			if (hypo[i][j]==1)
			{
				temp_indx.push_back(j);
			}
		}
		temp_t_id.push_back(temp_indx[0]);
		if (temp_indx.size()==2)
		{
			temp_t_id.push_back(temp_indx[1]-N1);
			parent[temp_indx[1]-N1]=temp_indx[0];
		}
		else if (temp_indx.size()==3)
		{
			temp_t_id.push_back(temp_indx[1]-N1);
			temp_t_id.push_back(temp_indx[2]-N1);
			parent[temp_indx[1]-N1]=temp_indx[0];
			parent[temp_indx[2]-N1]=temp_indx[0];
		}

		t_id.push_back(temp_t_id);
	}
			

}

int _tmain(int argc, _TCHAR* argv[])
{
	 CFileFind finder;
	 string img_dirPath;
	 img_dirPath="S:/GSoC_2015_Implementation/Registered_images/";
	 //"C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/Registered_images/"
	 //Reading number of files in the directory
	 int num_img=0;
	 DIR *dir;
	 struct dirent *ent;
	 // C:\\Users\\Sayan\\Dropbox\\GSoC_2015_Implementation\\Registered_images\\

	 if ((dir = opendir ("S:\\GSoC_2015_Implementation\\Registered_images\\")) != NULL) {
		/* print all the files and directories within directory */
		 while ((ent = readdir (dir)) != NULL) {
		 printf ("%s\n", ent->d_name);
		 num_img=num_img+1;
		 }
		 closedir (dir);
	 } else {
		/* could not open directory */
		 perror ("");
		 return EXIT_FAILURE;
	 }
	 num_img=num_img-2;//First two entries are path
	 //Uptohere
	 CString dirPath_1,dirPathC,dirPathC1;
	 CString cstr,cstrf;	 	 	 
	 dirPathC1 = img_dirPath.c_str();
	 dirPathC = dirPathC1 + "\\*.tif";
	 BOOL bWorking = finder.FindFile(dirPathC);
	 int count_of_slides=0;
	 cv::Mat mydialate1;
	 cv::Mat mydialate2;
	 Size img_size;
	 Size regstr_img_size;
	 //To save output of generate_tracklet
	 vector<vector<vector<int>>> TT;
	 TT.resize(num_img/2);
	 vector<vector<vector<double>>> vessel_vec;
	 vessel_vec.resize(num_img);
	 vector<vector<int>> parent_ids;
	 parent_ids.resize(num_img/2);
	 vector<vector<vector<Point>>> bounds_local;
	 bounds_local.resize(num_img);
	 vector<vector<double>> areas_local;
	 areas_local.resize(num_img);
	 vector<vector<Point>> centroids_local;
	 centroids_local.resize(num_img);
	 vector<vector<vector <double>>> features;
	 features.resize(num_img);
	 vector<Vec4i> hierarchy;
	 while (bWorking)

	{
		//Read the registered image
		CString dirPathC2;
		dirPathC2="S:/GSoC_2015_Implementation/Registered_images/";//"C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/Registered_images/"
		dirPath_1="S:/GSoC_2015_Implementation/double_seg_file/";//C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/double_seg_file/
		char *Save_images= "C:\\Users\\Sayan\\Dropbox\\GSoC_2015_Implementation\\Read_and_plot\\";
		bWorking = finder.FindNextFile();
		cstr = finder.GetFileName();
		CStringA cstr1(cstr);
		CString finpath=dirPathC1+cstr;
		_tprintf_s(_T("%s\n"), (LPCTSTR) finpath);
		_tprintf_s(_T("%s\n"), (LPCTSTR) cstr);
		//Variable to maintain the pairwise slide count ****Create a variable with the number of 1st slide to save the local bi-slide assosiation
		CString field;
		int index = 0;
		int img_inx;
		while (AfxExtractSubString(field,cstr,index,_T('.')))
		{
			index=index+1;
			if (index==2)
				img_inx=_ttoi(field);			
		}

		count_of_slides=count_of_slides+1;
		cout<<"count_of_slides::"<<count_of_slides<<"::"<<count_of_slides%2<<endl;
		cstrf=cstr;
		cstrf.Delete(cstrf.GetLength()-3,3);
		dirPath_1=dirPath_1+cstrf;
		CT2CA pszConvertedAnsiString1 (dirPath_1);	 
		string seg_file(pszConvertedAnsiString1);
		seg_file.append("bmp.mat");
		char *seg_file1=&seg_file[0];
		dirPathC2=dirPathC2+cstr;
	    CT2CA pszConvertedAnsiString (dirPathC2);
	    string image_name1(pszConvertedAnsiString);

		//void generate_tracklet can be started from here::
		
		if (count_of_slides%2==1){
			cout<<"First slide"<<endl;
			char *imagename=&image_name1[0];
			cv::Mat mydialate;
			read_and_dialate(imagename, seg_file1,  mydialate);
			mydialate1=mydialate;
			cv::Mat imgGray;
			cv::Mat img = imread(imagename);
			Size s = img.size();
			regstr_img_size=img.size();
			cout<<"Image Height:"<<s.height<<endl;
			cout<<"Image Width:"<<s.width<<endl;
			Size size(s.height/2,s.width/2);
			img_size=size;
			
					
		}

		else if(count_of_slides%2==0){
			cout<<"Second slide"<<endl;
			char *imagename2=&image_name1[0];
			cv::Mat mydialate_;
			read_and_dialate(imagename2, seg_file1,  mydialate_);
			mydialate2=mydialate_;
			cv::Mat imgGray;
			cv::Mat img = imread(imagename2);
			Size s = img.size();
			cout<<"Image Height:"<<s.height<<endl;
			cout<<"Image Width:"<<s.width<<endl;
			Size size(s.height/2,s.width/2);
			
		}

		//Here we will take the above two slides data to perform Bi-slide (local)

		if(count_of_slides%2==0){
			//Display the pair
			cv::Mat img2;
			resize(mydialate1, img2, img_size);//50% redeuction in display to fit in display-view
			imshow("image1", img2);	
			cv::Mat img3;
			resize(mydialate2, img3, img_size);//50% redeuction in display to fit in display-view
			imshow("image2", img3);	
			waitKey(5);
			char *imagename="current_image";
			if(img2.empty())
			{
				fprintf(stderr, "Can not load image %s\n", imagename);
				return -1;
			}

			vector<vector<Point>> blobs_1;
			vector<double> bolbarea_1;
			vector<Point> bolbcenter_1;			
			get_boundary_centroid( mydialate1,blobs_1, bolbarea_1,bolbcenter_1,hierarchy);
			bounds_local[img_inx-1]=blobs_1;
			areas_local[img_inx-1]=bolbarea_1;
			centroids_local[img_inx-1]=bolbcenter_1;
			vector<vector<Point>> blobs_2;
			vector<double> bolbarea_2;
			vector<Point> bolbcenter_2;
			get_boundary_centroid( mydialate2,blobs_2, bolbarea_2,bolbcenter_2,hierarchy);
			bounds_local[img_inx]=blobs_2;
			areas_local[img_inx]=bolbarea_2;
			centroids_local[img_inx]=bolbcenter_2;

			vector<vector <double>> Fsd_slide_1;
			for (int i = 0; i < blobs_1.size(); i++ ) {
				std::vector<double> X, Y;
				for(int j=0; j<blobs_1[i].size(); j++){
					X.push_back(blobs_1[i][j].x);
					Y.push_back(blobs_1[i][j].y);
					}
				vector<double> Fsd;
				get_FSDs(X,Y, Fsd);
				Fsd_slide_1.push_back(Fsd);
				features[img_inx-1]=Fsd_slide_1;
				}

			vector<vector <double>> Fsd_slide_2;
			for (int i = 0; i < blobs_2.size(); i++ ) {
				std::vector<double> X1, Y1;
				for(int j=0; j<blobs_2[i].size(); j++){
					X1.push_back(blobs_2[i][j].x);
					Y1.push_back(blobs_2[i][j].y);
					}
				vector<double> Fsd1;
				get_FSDs(X1,Y1, Fsd1);
				Fsd_slide_2.push_back(Fsd1);
				features[img_inx]=Fsd_slide_2;
				}
			

			vector<double> p_likelihood;
			vector<vector<int>> C_coeff;

			for (int i = 0; i < blobs_1.size(); i++ ) 
			{
				int obj_inx=i;
				int problem_id=1;
				vector<double> tmp_p_likelihood;
				vector<vector<int>> tmp_C_coeff;
				//dummy var
				vector<double> pre_vec_j(2,1);		
				vector<vector<double>> vec3;
				//
				calculate_likelihood(obj_inx,Fsd_slide_1,bolbcenter_1,bolbarea_1,Fsd_slide_2,bolbcenter_2,bolbarea_2,problem_id,blobs_2,pre_vec_j,vec3,tmp_p_likelihood,tmp_C_coeff);				
				//***Store tmp_p_likelihood in p_likelihood
				p_likelihood.insert(p_likelihood.end(), tmp_p_likelihood.begin(), tmp_p_likelihood.end());
				//***Store tmp_C_coeff in C_coeff
				C_coeff.insert(C_coeff.end(), tmp_C_coeff.begin(), tmp_C_coeff.end());
											
			}
			cout<<C_coeff[0][0]<<endl;
			cout<<C_coeff[1][0]<<endl;
			cout<<C_coeff[2][0]<<endl;
			cout<<C_coeff[13][1]<<endl;
			/*
			auto biggest = std::max_element(std::begin(p_likelihood), std::end(p_likelihood));
			std::cout << "Max element is " << *biggest
			<< " at position " << std::distance(std::begin(p_likelihood), biggest) << std::endl;
			*/
			//binary-integer-programming
			//test the cbc library
			OsiClpSolverInterface* si = new OsiClpSolverInterface();

			int n_cols =p_likelihood.size();//130
			double * objective    = new double[n_cols];//the objective coefficients
			double * col_lb       = new double[n_cols];//the column lower bounds
			double * col_ub       = new double[n_cols];//the column upper bounds

			for (int i = 0; i < n_cols; i++ ) 
			{
				objective[i]=p_likelihood[i];
				col_lb[i] = 0.0;
				col_ub[i] = si->getInfinity();//1.0;
			}
			int N1=bolbcenter_1.size();
			int N2=bolbcenter_2.size();
			int n_rows = N1+N2;//1075
			double * row_lb = new double[n_rows]; 
			double * row_ub = new double[n_rows];
			//Define the constraint matrix.
			CoinPackedMatrix * matrix =  new CoinPackedMatrix(false,0,0);
			matrix->setDimensions(0, n_cols);
			for (int i = 0; i < n_rows; i++ ) 
			{
				CoinPackedVector rowins;
				for (int j = 0; j < n_cols; j++ ) 
				{
					rowins.insert(j,C_coeff[j][i]);
					
				}	
				row_lb[i] = 0;//-1.0 * si->getInfinity();
				row_ub[i] =1.0;
				matrix->appendRow(rowins);
			}
			//load the problem to OSI
			si->loadProblem(*matrix, col_lb, col_ub, objective, row_lb, row_ub);
			si->setObjSense(-1);			
			for (int i = 0; i < n_cols; i++ ) 
			{
				si->setInteger(i);
			}
			si->branchAndBound();
			const double * solution = si->getColSolution();
			std::vector<double> vsol(solution, solution+n_cols );
			cout<<vectorsum(vsol)<<endl;
			const double objective_value = si->getObjValue();

			 //free the memory
			if(objective != 0)   { delete [] objective; objective = 0; }
			if(col_lb != 0)      { delete [] col_lb; col_lb = 0; }
			if(col_ub != 0)      { delete [] col_ub; col_ub = 0; }
			if(row_lb != 0)      { delete [] row_lb; row_lb = 0; }
			if(row_ub != 0)      { delete [] row_ub; row_ub = 0; }
			if(matrix != 0)      { delete matrix; matrix = 0; }
			//max(N1,N2)
			vector<int> parent(N2, std::numeric_limits<double>::quiet_NaN());//might need to initialized all elements containing Nan 
			vector<vector<int>> t_id;
			get_matching_result(vsol,C_coeff,Fsd_slide_1,Fsd_slide_2,bolbcenter_1,bolbcenter_2,parent,t_id);
			// get vessel vectors
			
			vector<vector<double>> vec1;
			for(int i = 0; i < bolbcenter_1.size(); i++ ) {
				vector<double> vec_tmp_1(2,std::numeric_limits<double>::quiet_NaN());
				vec1.push_back(vec_tmp_1);
			}

			vector<vector<double>> vec2;
			for(int i = 0; i < bolbcenter_2.size(); i++ ) {
				vector<double> vec_tmp_1(2,std::numeric_limits<double>::quiet_NaN());
				vec2.push_back(vec_tmp_1);
			}
						
			for(int i = 0; i < t_id.size(); i++ ) 
			{
				int id1 = t_id[i][0];
				vector<int> id2(t_id[i].begin() + 1, t_id[i].end());
				if (id2.size()>1)
				{
					double sum_x = 0;
					double sum_y = 0;
					for(int j = 0; j < id2.size(); j++ ) 
					{
						Point vec_tmp=bolbcenter_2[id2[j]]-bolbcenter_1[id1];
						vec2[id2[j]][0]=vec_tmp.x;
						vec2[id2[j]][1]=vec_tmp.y;
						sum_x = sum_x + vec_tmp.x;
						sum_y = sum_y + vec_tmp.y;
					}
					vec1[id1][0]=sum_x/id2.size();
					vec1[id1][1]=sum_y/id2.size();
				}
				else if (id2.size()==1)
				{
					Point vec_tmp=bolbcenter_2[id2[0]]-bolbcenter_1[id1];
					vec2[id2[0]][0]=vec_tmp.x;
					vec2[id2[0]][1]=vec_tmp.y;
					vec1[id1][0]=vec_tmp.x;
					vec1[id1][1]=vec_tmp.y;

				}


			}
			//save some vector before starting "generate_structure"
			TT[(img_inx-1)/2]=t_id;
			vessel_vec[img_inx-1]=vec1;
			vessel_vec[img_inx]=vec2;
			parent_ids[(img_inx-1)/2]=parent;
		}
		
	
	}

	//void generate_structure ()
	vector<vector<vector<int>>> step2_TT;
	step2_TT.resize((num_img/2)-1);
	vector<vector<int>> step2_parent_ids;
	step2_parent_ids.resize((num_img/2)-1);
	
	for(int i = 0; i < TT.size()-1; i++ ) 
	{
		vector<vector<int>> pair1=TT[i+1];
		vector<vector<int>> pair2=TT[i+1];

		vector<vector<int>> frame2; //Can have zero, one or more than one component 
		for(int j = 0; j < pair1.size(); j++ ) 
		{
			vector<int> temp_p1=pair1[j];
			
			if (temp_p1.size()==1)
			{
				vector<int> temp_p3(1, std::numeric_limits<double>::quiet_NaN());
				frame2.push_back(temp_p3);				
			}
			else 
			{
				vector<int> temp_p2;
				for(int k = 0; k < temp_p1.size()-1; k++ )
				{
					temp_p2.push_back(temp_p1[k+1]);
				}
				frame2.push_back(temp_p2);				
			}
						
		}
		vector<vector<Point>> bound2;
		bound2=bounds_local[2*(i+1)-1];
		vector<Point> centroid_vec2;
		centroid_vec2=centroids_local[2*(i+1)-1];
		vector<vector <double>> feature_vec2;
		feature_vec2=features[2*(i+1)-1];
		vector<double> area_f2;
		area_f2=areas_local[2*(i+1)-1];
		vector<vector<double>> vec2_2;
		vec2_2=vessel_vec[2*(i+1)-1];

		vector<int> frame3; //Only one component always
		for(int j = 0; j < pair2.size(); j++ ) 
		{
			frame3.push_back(pair2[j][0]);
		}
		vector<vector<Point>> bound3;
		bound3=bounds_local[2*(i+1)];
		vector<Point> centroid_vec3;
		centroid_vec3=centroids_local[2*(i+1)];
		vector<vector <double>> feature_vec3;
		feature_vec3=features[2*(i+1)];
		vector<double> area_f3;
		area_f3=areas_local[2*(i+1)];
		vector<vector<double>> vec3_2;
		vec3_2=vessel_vec[2*(i+1)];

		vector<double> p_likelihood_2;
		vector<vector<int>> C_coeff_2;
		for (int j = 0; j < bound2.size(); j++ ) 
		{
			vector<double> pre_vec_j=vec2_2[j];
			
			if ((isnan(pre_vec_j[0]))&&(isnan(pre_vec_j[1])))
			{
				pre_vec_j[0]=1;
				pre_vec_j[1]=1;
			}
			int obj_inx=j;
			int problem_id=2;
			vector<double> tmp_p_likelihood;
			vector<vector<int>> tmp_C_coeff;
			calculate_likelihood(obj_inx,feature_vec2,centroid_vec2,area_f2,feature_vec3,centroid_vec3,area_f3,problem_id,bound3,pre_vec_j,vec3_2,tmp_p_likelihood,tmp_C_coeff);				
			//***Store tmp_p_likelihood in p_likelihood
			p_likelihood_2.insert(p_likelihood_2.end(), tmp_p_likelihood.begin(), tmp_p_likelihood.end());
			//***Store tmp_C_coeff in C_coeff
			C_coeff_2.insert(C_coeff_2.end(), tmp_C_coeff.begin(), tmp_C_coeff.end());
											
		}

		OsiClpSolverInterface* si = new OsiClpSolverInterface();

		int n_cols =p_likelihood_2.size();//130
		double * objective    = new double[n_cols];//the objective coefficients
		double * col_lb       = new double[n_cols];//the column lower bounds
		double * col_ub       = new double[n_cols];//the column upper bounds

		for (int i = 0; i < n_cols; i++ ) 
		{
			objective[i]=p_likelihood_2[i];
			col_lb[i] = 0.0;
			col_ub[i] = si->getInfinity();//1.0;
		}
		int N1=centroid_vec2.size();
		int N2=centroid_vec3.size();
		int n_rows = N1+N2;//1075
		double * row_lb = new double[n_rows]; 
		double * row_ub = new double[n_rows];
		//Define the constraint matrix.
		CoinPackedMatrix * matrix =  new CoinPackedMatrix(false,0,0);
		matrix->setDimensions(0, n_cols);
		for (int i = 0; i < n_rows; i++ ) 
		{
			CoinPackedVector rowins;
			for (int j = 0; j < n_cols; j++ ) 
			{
				rowins.insert(j,C_coeff_2[j][i]);
					
			}	
			row_lb[i] = 0;//-1.0 * si->getInfinity();
			row_ub[i] =1.0;
			matrix->appendRow(rowins);
		}
		//load the problem to OSI
		si->loadProblem(*matrix, col_lb, col_ub, objective, row_lb, row_ub);
		si->setObjSense(-1);			
		for (int i = 0; i < n_cols; i++ ) 
		{
			si->setInteger(i);
		}
		si->branchAndBound();
		const double * solution = si->getColSolution();
		std::vector<double> vsol(solution, solution+n_cols );
		cout<<vectorsum(vsol)<<endl;
		const double objective_value = si->getObjValue();

			//free the memory
		if(objective != 0)   { delete [] objective; objective = 0; }
		if(col_lb != 0)      { delete [] col_lb; col_lb = 0; }
		if(col_ub != 0)      { delete [] col_ub; col_ub = 0; }
		if(row_lb != 0)      { delete [] row_lb; row_lb = 0; }
		if(row_ub != 0)      { delete [] row_ub; row_ub = 0; }
		if(matrix != 0)      { delete matrix; matrix = 0; }

		vector<int> parent_2(N2, std::numeric_limits<double>::quiet_NaN());//might need to initialized all elements containing Nan 
		vector<vector<int>> t_id_2;
		get_matching_result(vsol,C_coeff_2,feature_vec2,feature_vec3,centroid_vec2,centroid_vec3,parent_2,t_id_2);

		step2_TT[i]=t_id_2;
		step2_parent_ids[i]=parent_2;

		
	}
	//end of void generate_structure ()

	//void generate_3dvessel ()
	vector<vector<int>> parents;
	int in1=0;
	for (in1; in1 < step2_parent_ids.size(); in1++ ) 
		{
			parents.push_back(parent_ids[in1]);
			parents.push_back(step2_parent_ids[in1]);
		}
	parents.push_back(parent_ids[in1]);
	vector<vector<vector<Point>>> vessels_bounds;
	vector<vector<int>> vessels_ids;
	int in2 = parents.size()-1;
	for(; in2 >= 0; in2--) 
		{
			vector<int> current_frame_parent;
			current_frame_parent=parents[in2];
			vector<vector<Point>> current_bound;
			current_bound=bounds_local[in2+1];
			for (int j = 0; j < current_frame_parent.size(); j++ ) 
			{
				int parent_id=current_frame_parent[j];
				cout<<isnan(double(parent_id))<<endl;
				
				if (parent_id!=-2147483648)//isnan not working
				{
					vector<int> vessel(bounds_local.size(), 0);//std::numeric_limits<double>::quiet_NaN());//'0's are initialized in Matlab for C++ '0' being valid index
					vessel[in2+1] = j+1;//***(0 indexing) added +1
					vector<vector<Point>> vessel_bound;
					vessel_bound.resize(bounds_local.size());
					vessel_bound[in2+1]=current_bound[j];
					int k = in2;
					for(; k >= 1; k--) 
					{
						if (parent_id!=-2147483648)//isnan not working
						{
							vector<vector<Point>> temp_bound;
							temp_bound=bounds_local[k];
							vessel_bound[k]=temp_bound[parent_id];
							vessel[k] = parent_id+1;//***(0 indexing) added +1
							int tmp_id = parent_id;
							vector<int> tmp_parent=parents[k-1];
							parent_id=tmp_parent[tmp_id];

						}
						else
						{
							k=0;
						}
					}
					if ((k==1)&&(parent_id!=-2147483648))//isnan not working
					{
						vector<vector<Point>> temp_bound;
						temp_bound=bounds_local[k-1];
						vessel_bound[k-1]=temp_bound[parent_id];
						vessel[k-1] = parent_id+1;//***(0 indexing) added +1
					}
					vessels_ids.push_back(vessel);//***(0 indexing) added +1
					vessels_bounds.push_back(vessel_bound);

				}

			}
    
		}
	// sort the ids and the corresponding bounds 
	cv::Mat vessel_mat= cv::Mat(vessels_ids.size(), vessels_ids.at(0).size(), CV_64FC1);
	for(int i=0; i<vessel_mat.rows; ++i)
	{
     for(int j=0; j<vessel_mat.cols; ++j)
	 {
          vessel_mat.at<double>(i, j) = double(vessels_ids.at(i).at(j));
	 }
	}
	//get first column
	cv::Mat col_one= vessel_mat.col(1);//***** Need to change to 0, put 1 as the first column are all 0 elements
	//sort the first column and save indices in dst
	cv::Mat1i test_idx;
	cv::sortIdx(col_one, test_idx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
	vector<int> idx;
	test_idx.copyTo(idx);
	// now build your final matrix "vector<vector>"
	vector<vector<int>> vessel_mat_sorted(vessels_ids.size());
	for(int y = 0; y < vessel_mat.rows; y++){
	   vessel_mat_sorted.at(idx.at(y))=vessels_ids.at(y);	  //***vessel_mat_sorted=>(0 indexing) added +1 
	}
	// Destroy Mat objects
	vessel_mat.release();
	col_one.release();
	test_idx.release();
	
	// sort corresponding bounds
	vector<vector<vector<Point>>> vessels_sorted_bounds(vessels_bounds.size());
	for(int y = 0; y < idx.size(); y++){
	   vessels_sorted_bounds.at(idx.at(y))=vessels_bounds.at(y);	   
	}

	vector<vector<vector<Point>>> selected_vessel_bounds;
	vector<vector<int>> selected_vessel_mat;
    
	for(int i=0; i<vessel_mat_sorted.size(); ++i)
	{
		int non_zero_count = count_if (vessel_mat_sorted.at(i).begin(), vessel_mat_sorted.at(i).end(), IsNonZero);
		if(double(non_zero_count)>=0.3*vessel_mat_sorted.at(0).size())
		{
			vector<int> a=vessel_mat_sorted.at(i);//***vessel_mat_sorted=>(0 indexing) added +1 
			int j=i+1;
			while (j<vessel_mat_sorted.size())
			{
				vector<int> b=vessel_mat_sorted.at(j);//***vessel_mat_sorted=>(0 indexing) added +1 
				vector<int> aa;
				vector<int> bb;
				aa=nonzeroitems_vec(a);
				bb=nonzeroitems_vec(b);
				int testcommon=CheckCommon(aa,bb);
				if (testcommon!=0)
				{
					j=vessel_mat_sorted.size()+100;
				}
				j=j+1;
			}
			if(j!=vessel_mat_sorted.size()+100)//a is a good vessel
			{
				selected_vessel_bounds.push_back(vessels_sorted_bounds.at(i));
				selected_vessel_mat.push_back(a);//Might consider to substract '1' for each entry of 'a'...
			}

		}
		
	}

	//end of void generate_3dvessel ()

	//generate_mesh()
	vector<vector<vector<vector<double>>>> selected_interpolated_bounds;
	
	for(int i=0; i<selected_vessel_bounds.size(); ++i)
	{
		cout<<i<<endl;
		vector<vector<Point>> vessel_bounds;
		vessel_bounds=selected_vessel_bounds.at(i);
		vector<vector<vector<double>>> Intrpltd_bound;
		//spline_interpolation() *call within generate_mesh()

		vector<float> L ; // the perimeter of each object
		vector<int> Sz; // the # of points in the boundary of each object
		vector<vector<float>> C; // the centroid of each object
		for(int j=0; j<vessel_bounds.size(); ++j)
		{
			vector<Point> bound_obj=vessel_bounds.at(j);
			if (bound_obj.size()>0)
			{
			
				vector<Point> diff_temp;
				//write adjacent_difference for cv:Point object
				vector<Point> append_bound_obj=bound_obj;
				append_bound_obj.push_back(bound_obj.at(0));
				float L_temp=0;
				float tmp11=0,tmp22=0;
				for(int k=0; k<append_bound_obj.size()-1; ++k)
				{
					float tmp1=append_bound_obj.at(k+1).x-append_bound_obj.at(k).x;
					float tmp2=append_bound_obj.at(k+1).y-append_bound_obj.at(k).y;	
					tmp11=float(append_bound_obj.at(k).x)+tmp11;
					tmp22=float(append_bound_obj.at(k).y)+tmp22;
					diff_temp.push_back(cv::Point(append_bound_obj.at(k+1).x-append_bound_obj.at(k).x,append_bound_obj.at(k+1).y-append_bound_obj.at(k).y));
					L_temp=L_temp+ sqrt((tmp1*tmp1)+(tmp2*tmp2));					
				}
				L.push_back(L_temp);
				Sz.push_back(bound_obj.size());
				vector<float> tempC;
				tempC.push_back(tmp11/bound_obj.size());
				tempC.push_back(tmp22/bound_obj.size());
				C.push_back(tempC);
			}
			else
			{
				L.push_back(0);
				Sz.push_back(0);
				vector<float> tempC;
				tempC.push_back(0);
				tempC.push_back(0);
				C.push_back(tempC);
			}
		}

		sort( Sz.begin(), Sz.end() );
		Sz.erase( unique( Sz.begin(), Sz.end() ), Sz.end());
		int M=Sz.at(1);
		vector<float> delta_L=L;
		std::transform(delta_L.begin(), delta_L.end(), delta_L.begin(), std::bind1st(std::multiplies<float>(),(1/float(M))));		
		double minsize = std::numeric_limits<int>::max();

		vector<vector<double>> sampleX;
		vector<vector<double>> sampleY;

		for(int j=0; j<vessel_bounds.size(); ++j)
		{
			vector<Point> bound_obj=vessel_bounds.at(j);
			if (bound_obj.size()>0)
			{
				//getptsOneSlice(bound_obj,delta_L.at(i),L.at(i));
				vector<int> temp1;
				vector<int> temp2;
				for(int k=0; k< bound_obj.size(); ++k)
				{
					temp1.push_back(int(bound_obj.at(k).x));
					temp2.push_back(int(bound_obj.at(k).y));
				}
				sort( temp1.begin(), temp1.end() );
				int maxval=temp1.at(0);
				vector<int> temp3=temp2;
				sort( temp3.begin(), temp3.end() );
				vector<int> y;
				std::vector<size_t> y1(temp2.size());
				std::iota(y1.begin(), y1.end(), 0);
				std::copy_if(y1.begin(), y1.end(), std::back_inserter(y), [=](size_t inx) { return temp2[inx] == temp3[0]; });	
				int idx = 0;
				for(int k=0; k< y.size(); ++k)
				{
					if(int(bound_obj.at(y[k]).x)>maxval)
					{
						maxval=int(bound_obj.at(y[k]).x);
						idx=y[k];
					}
				}
				if ((bound_obj[0].x==bound_obj[bound_obj.size()-1].x)&(bound_obj[0].y==bound_obj[bound_obj.size()-1].y))
				{
					bound_obj.erase(bound_obj.begin());
				}
				vector<Point> res;
				for(int k=idx;k<bound_obj.size();k++)
				{
					res.push_back(bound_obj[k]);
				}
				for(int k=0;k<idx;k++)
				{
					res.push_back(bound_obj[k]);
				}
				vector<Point> append_res=res;
				vector<double> resDiff;
				append_res.push_back(res.at(0));
				double resDiff_temp=0;
				vector<double> res1;
				vector<double> res2;
				for(int k=0; k<append_res.size()-1; ++k)
				{
					float tmp1=append_res.at(k+1).x-append_res.at(k).x;
					float tmp2=append_res.at(k+1).y-append_res.at(k).y;	
					resDiff_temp=resDiff_temp+sqrt(pow(tmp1,2)+pow(tmp2,2));
					resDiff.push_back(resDiff_temp);	
					res1.push_back(double(res[k].x));
					res2.push_back(double(res[k].y));
				}

				double maxL=max(double(L.at(j)),resDiff.at(resDiff.size()-1));
				vector<double> XX=resDiff;
				std::transform(XX.begin(), XX.end(), XX.begin(), std::bind1st(std::multiplies<double>(),(1/maxL)));	
				vector<double> XXq=linspace_intrvl(0,double(L.at(j)),double(delta_L.at(j)));
				std::transform(XXq.begin(), XXq.end(), XXq.begin(), std::bind1st(std::multiplies<double>(),(1/maxL)));	
				vector<double> XIi;
				vector<double> YIi;
				vector<double> XI;
				vector<double> YI;
				//***Alglib spline 
				alglib::real_1d_array AX, AY1, AY2;
				AX.setcontent(XX.size(), &(XX[0]));
				AY1.setcontent(res1.size(), &(res1[0]));
				AY2.setcontent(res2.size(), &(res2[0]));
				alglib::spline1dinterpolant spline1;
				alglib::spline1dbuildlinear(AX, AY1, XX.size(), spline1);
				alglib::spline1dinterpolant spline2;
				alglib::spline1dbuildlinear(AX, AY2, XX.size(), spline2);
				for(int k=0; k<XXq.size(); ++k)
				{
					XIi.push_back(alglib::spline1dcalc(spline1,XXq[k]));
					YIi.push_back(alglib::spline1dcalc(spline2,XXq[k]));
					if (isnan(alglib::spline1dcalc(spline1,XXq[k]))==0)
					{
						XI.push_back(alglib::spline1dcalc(spline1,XXq[k]));
					}
					if (isnan(alglib::spline1dcalc(spline2,XXq[k]))==0)
					{
						YI.push_back(alglib::spline1dcalc(spline2,XXq[k]));
					}
				}

				//getptsOneSlice ends return "XI" & "YI" ***just make sure the isnan working in case of isnan***
				sampleX.push_back(XI);
				sampleY.push_back(YI);
				minsize = min(minsize,double(XI.size()));

			}
		}

		vector<vector<double>> Xpts;
		vector<vector<double>> Ypts;

		for(int j=0; j<sampleX.size(); ++j)
		{
			vector<double> xi;
			xi=sampleX.at(j);
			vector<double> xi_inx=linspace_intrvl(0,double(xi.size()),double(floor(xi.size()/minsize)));
			vector<double> Xi;
			for(int k=0; k<xi_inx.size(); ++k)
			{
				Xi.push_back(xi.at(int(xi_inx[k])));
			}
			Xi.erase (Xi.begin()+minsize,Xi.end());
			Xpts.push_back(Xi);

			vector<double> yi;
			yi=sampleY.at(j);
			vector<double> yi_inx=linspace_intrvl(0,double(yi.size()),double(floor(yi.size()/minsize)));
			vector<double> Yi;
			for(int k=0; k<yi_inx.size(); ++k)
			{
				Yi.push_back(yi.at(int(yi_inx[k])));
			}
			Yi.erase (Yi.begin()+minsize,Yi.end());
			Ypts.push_back(Yi);
		}

		vector<vector<float>> C_append=C;
		C_append.push_back(C[0]);
		double distance=0;
		for(int j=0; j<C_append.size()-1; ++j)
		{
			double tmp1=C_append.at(j+1)[0]-C_append.at(j)[0];
			double tmp2=C_append.at(j+1)[1]-C_append.at(j)[1];	
			distance=distance+ sqrt((tmp1*tmp1)+(tmp2*tmp2));					
		}
		double total_frame = 100;
		double delta_slice = 0.02;

		double startFrame = 1;
		double endFrame = sampleX.size();

		vector<vector<double>> Xpts_t;
		Xpts_t=vec_transpose(Xpts);
		vector<vector<double>> Ypts_t;
		Ypts_t=vec_transpose(Ypts);
		vector<double> XX1=linspace_intrvl(startFrame,endFrame+1,1);
		vector<double> XXq1=linspace_intrvl(startFrame,endFrame+delta_slice,delta_slice);
		//***Alglib spline 
		vector<vector<double>> X_spline;
		vector<vector<double>> Y_spline;
		alglib::real_1d_array AX, AY1, AY2;
		AX.setcontent(XX1.size(), &(XX1[0]));
		for(int j=0; j<Xpts_t.size(); ++j)
		{
			vector<double> tmp_xpts=Xpts_t[j];
			vector<double> tmp_ypts=Ypts_t[j];
			AY1.setcontent(tmp_xpts.size(), &(tmp_xpts[0]));
			AY2.setcontent(tmp_ypts.size(), &(tmp_ypts[0]));
			alglib::spline1dinterpolant spline1;
			alglib::spline1dbuildlinear(AX, AY1, XX1.size(), spline1);
			alglib::spline1dinterpolant spline2;
			alglib::spline1dbuildlinear(AX, AY2, XX1.size(), spline2);
			vector<double> tmp_X_spline;
			vector<double> tmp_Y_spline;
			for(int k=0; k<XXq1.size(); ++k)
			{
				tmp_X_spline.push_back(alglib::spline1dcalc(spline1,XXq1[k]));
				tmp_Y_spline.push_back(alglib::spline1dcalc(spline2,XXq1[k]));
			}
			X_spline.push_back(tmp_X_spline);
			Y_spline.push_back(tmp_Y_spline);
		}
		vector<vector<double>> X_spline_t;
		X_spline_t=vec_transpose(X_spline);
		vector<vector<double>> Y_spline_t;
		Y_spline_t=vec_transpose(Y_spline);

		
		
		for(int j=0; j<X_spline_t.size(); ++j)
		{			
			vector<vector<double>> tmp_intrp_bnd1;
			for(int k=0; k<X_spline_t[0].size(); ++k)
			{
				vector<double> tmp_intrp_bnd;
				tmp_intrp_bnd.push_back(X_spline_t[j][k]);
				tmp_intrp_bnd.push_back(Y_spline_t[j][k]);
				tmp_intrp_bnd1.push_back(tmp_intrp_bnd);
			}
			Intrpltd_bound.push_back(tmp_intrp_bnd1);

		}
		
		//end of spline_interpolation() return Intrpltd_bound  
		selected_interpolated_bounds.push_back(Intrpltd_bound);

	}

	vector<vector<int>> selected_vessel_mat_t=vec_transpose_4_int(selected_vessel_mat);//selected_vessel_mat=>Might consider to substract '1' for each entry 

	//get first column
	vector<int> vessel_mat_col=selected_vessel_mat_t[1] ;//***** Need to change to 0, put 1 as the first column are all 0 elements
	//vessel_mat_col=>Might consider to substract '1' for each entry (**Debug**)
	vector<int> C_vessel_mat_col=vessel_mat_col ;
	//C_vessel_mat_col (unique( C_vessel_mat_col.begin(), C_vessel_mat_col.end() ));
	sort( C_vessel_mat_col.begin(), C_vessel_mat_col.end() );
	C_vessel_mat_col.erase( unique( C_vessel_mat_col.begin(), C_vessel_mat_col.end() ), C_vessel_mat_col.end());

	for(int i=0; i<C_vessel_mat_col.size(); ++i)
	{
		if (C_vessel_mat_col[i]!=0)
		{
			vector<int> y_;
			std::vector<size_t> y1_(vessel_mat_col.size());
			std::iota(y1_.begin(), y1_.end(), 0);
			std::copy_if(y1_.begin(), y1_.end(), std::back_inserter(y_), [=](size_t inx) { return vessel_mat_col[inx] == C_vessel_mat_col[i]; });	
			auto biggest = std::max_element(std::begin(y_), std::end(y_));	
			int smax=*biggest;// start index
			auto smallest = std::min_element(std::begin(y_), std::end(y_));
			int emin=*smallest;// end index
			if(smax==emin)
			{
				vector<vector<vector<double>>> one_vessel = selected_interpolated_bounds[smax];
				for(int j=0;j<one_vessel.size();j++)
				{
					vector<vector<double>> bound_one_vessel = one_vessel[j];
					vector<Point> InputArray1;
					//roipoly using the points vessel_obj and create a image to save
					for(int k=0;k<bound_one_vessel.size();k++)
					{
						//InputArray.push_back(cv::Point(vessel_obj[k][0],vessel_obj[k][1]));
						InputArray1.push_back(cv::Point(int(round(bound_one_vessel[k][1])),int(round(bound_one_vessel[k][0]))));
					}
					vector<vector<Point>> trytraj1;
					trytraj1.push_back(InputArray1);
					cv::Mat BW1 = cv::Mat::zeros(regstr_img_size.width,regstr_img_size.height, CV_8UC3);
					drawContours(BW1,trytraj1,0,Scalar(255, 255, 255),CV_FILLED,8);//,vector<Vec4i>(),0,Point()
					imshow("image5", BW1);	
					waitKey(5);
					std::ostringstream name;
					name << i <<"." << j << ".bspline.tif";
					cv::imwrite(name.str(), BW1);
					//Save the image in order
				}

			}
			else
			{
				int max_len = 0;
				for(int j=emin; j<=smax; ++j)
				{
					if(selected_interpolated_bounds[j].size()>max_len)
					{
						max_len=selected_interpolated_bounds[j].size();
					}
				}

				for(int p=0; p<max_len; ++p)// all the columns
				{
					//BW = zeros(size(I,1),size(I,2));//Create a black image "BW" (0 pixel of size of 'output_image.0.tif')
					cv::Mat BW = cv::Mat::zeros(regstr_img_size.width,regstr_img_size.height, CV_8UC3);
					imshow("image3", BW);	
					waitKey(5);
					for(int j=emin; j<=smax; ++j)//all the rows
					{
						vector<vector<vector<double>>> vessel_objs = selected_interpolated_bounds[j];
						if(p<vessel_objs.size())
						{
							vector<vector<double>> vessel_obj = vessel_objs[p];
							vector<Point> InputArray;
							//roipoly using the points vessel_obj and create a image to save
							for(int k=0;k<vessel_obj.size();k++)
							{
								//InputArray.push_back(cv::Point(vessel_obj[k][0],vessel_obj[k][1]));
								InputArray.push_back(cv::Point(int(round(vessel_obj[k][1])),int(round(vessel_obj[k][0]))));
							}
							vector<vector<Point>> trytraj;
							trytraj.push_back(InputArray);
							cv::Mat BWi = cv::Mat::zeros(regstr_img_size.width,regstr_img_size.height, CV_8UC3);
							drawContours(BWi,trytraj,0,Scalar(255, 255, 255),CV_FILLED,8);//,vector<Vec4i>(),0,Point()
							BW=BW|BWi;
							imshow("image4", BWi);	
							waitKey(5);
							std::ostringstream name;
							name << i <<"." << j << ".bspline.tif";
							cv::imwrite(name.str(), BWi);

							//Save the image in order (Need to debug order)loop in 1817, 1823
						}
					}
					//Save the image in order (*Might be here)
				}

			}
		}
	}



	 return 0;
}