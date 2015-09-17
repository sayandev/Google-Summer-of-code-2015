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
#include <cstddef>      
#include <cmath>        
#include <limits>
#include <valarray> 
#include <complex>
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
					}
			}
		}
	    cv::dilate(mat_dst, img341, getStructuringElement(MORPH_RECT, Size (5,5)));//Perform Image dialation on the RGB image			

	}

void get_boundary_centroid(cv::Mat dialated,vector<vector<Point>> &blobs_sorted, vector<double> &bolb_area_sorted,vector<Point> &bolb_center1_sorted)
{

			int top=30;
			//comment if not returning the sorted values
			vector<vector<Point>> blobs;
			vector<double> bolb_area;
			vector<Point> bolb_center1;

			//Uncomment if not returning the sorted values
			//vector<Point> bolb_center1_sorted;
			//vector<vector<Point>> blobs_sorted;

			cv::Mat img225;
			dialated.convertTo(img225,CV_8U);
			findContours(img225,blobs,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);//*** Find contour matches with Matlab better then labelBolbs
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

			for (int i = 0; i < bolb_area.size(); i++ ) {/
				bolb_center1_sorted.push_back(bolb_center1.at(vp[i].second));//*** Better match with Matlab version
				//bolb_center1_sorted.push_back(bolb_center1.at(tmp_inx[i]));
				
				}
			
			for (int i = 0; i < top; i++ ) {
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
	vector<Point> blobs_intrp;
	for ( int j = 0; j < eq_spaced_x.size(); ++j)
	{
		eq_spaced_x.at(j)=X.at(tbins.at(j)-1)+(X.at(tbins.at(j))-X.at(tbins.at(j)-1))*s1.at(j);
		eq_spaced_y.at(j)=Y.at(tbins.at(j)-1)+(Y.at(tbins.at(j))-Y.at(tbins.at(j)-1))*s1.at(j);
		blobs_intrp.push_back(Point(eq_spaced_x.at(j),eq_spaced_y.at(j)));
		
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
	for (int i = 0; i < num_intrp-1; ++i)
    {
		std::complex<double> temp_comp;
		temp_comp=(fftarr[i])*conj(fftarr[i]);
		fX.at(i)=temp_comp.real();
    }
	vector<double> obj_fsd(num_intrp-2);
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
	if(omega3!=0)
	{

		if(type==1)
		{
			omega1 = 8 * omega3;
			omega2 = 4 * omega3;
		}

		else
		{
			omega1 = 6 * omega3;
			omega2 = 4 * omega3;
		}
		
	}
	else
	{

		if(type==1)
		{
			omega1 = 5;
			omega2 = 3;
		}

		else
		{
			omega1 = 6;
			omega2 = 4;
		}
		

	}

	double omegas[3]={omega1/(omega1+omega2+omega3),omega2/(omega1+omega2+omega3),omega3/(omega1+omega2+omega3)};
					

	if(problem_id==2)
	{
		if(type==1)
		{
			like_values=omegas[0]*exp(-diff_feature/delta1)+omegas[1]*exp(-diff_centroid/delta2)+omegas[2]*exp(acos(cos_theta1)-acos(cos_theta2));
		}

		else
		{
			like_values=omegas[0]*exp(-diff_feature/delta1)+omegas[1]*exp(-diff_centroid/delta2)+omegas[2]*exp(acos(cos_theta1)-acos(cos_theta2));
		}
	}
	else
	{
		if(type==1)
		{
			like_values=omegas[0]*exp(-diff_feature/delta1)+omegas[1]*exp(-diff_centroid/delta2);
		}

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
				vector<double>::const_iterator xi;
				vector<double>::const_iterator yi;
				for (xi=disp_boundary1_x.begin(), yi=disp_boundary1_y.begin(); xi!=disp_boundary1_x.end(); ++xi, ++yi)
				{
					append(bound1, make<point_2d>(*xi, *yi));
				}
							
			}
			correct(bound1);
			polygon_2d bound2;
			{						
				vector<double>::const_iterator xi;
				vector<double>::const_iterator yi;
				for (xi=disp_boundary2_x.begin(), yi=disp_boundary2_y.begin(); xi!=disp_boundary2_x.end(); ++xi, ++yi)
				{
					append(bound2, make<point_2d>(*xi, *yi));
				}
							
			}
			correct(bound2);

			std::deque<polygon> intersect;
			try
			{
				boost::geometry::intersection(bound1, bound2, intersect);
			}catch(const boost::geometry::overlay_invalid_input_exception){

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
					break;
				}


				
				int iuni = 0;
				std::vector< double > com_boundary_x;
				std::vector< double > com_boundary_y;
				
				double poly_area=0;
				BOOST_FOREACH(polygon const& p2, unionpoly)
				{
					poly_area=boost::geometry::area(p2);
					
					
					for (auto it2 = boost::begin(boost::geometry::exterior_ring(p2)); it2 != boost::end(boost::geometry::exterior_ring(p2)); ++it2)
						{

							com_boundary_x.push_back(get<0>(*it2));

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
	

	normalizevector(p_likelihood,norm_p_likelihood);

}

void get_matching_result(vector<double> vsol,vector<vector<int>> C_coeff,vector<vector <double>> Fsd_slide_1,vector<vector <double>> Fsd_slide_2,vector<Point> bolbcenter_1,vector<Point> bolbcenter_2, vector<int> &parent,vector<vector<int>> &t_id)
{
	//find selected hypothesis
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
	 img_dirPath="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/Registered_images/";
	 //Reading number of files in the directory
	 int num_img=0;
	 DIR *dir;
	 struct dirent *ent;
	 if ((dir = opendir (" C:\\Users\\Sayan\\Dropbox\\GSoC_2015_Implementation\\Registered_images\\")) != NULL) {
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
	 while (bWorking)

	{
		//Read the registered image
		CString dirPathC2;
		dirPathC2="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/Registered_images/";
		dirPath_1="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/double_seg_file/";
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
			get_boundary_centroid( mydialate1,blobs_1, bolbarea_1,bolbcenter_1);
			bounds_local[img_inx-1]=blobs_1;
			areas_local[img_inx-1]=bolbarea_1;
			centroids_local[img_inx-1]=bolbcenter_1;

			vector<vector<Point>> blobs_2;
			vector<double> bolbarea_2;
			vector<Point> bolbcenter_2;
			get_boundary_centroid( mydialate2,blobs_2, bolbarea_2,bolbcenter_2);
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
				col_ub[i] = si->getInfinity();
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
				row_lb[i] = 0;
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

			vector<int> parent(N2, std::numeric_limits<double>::quiet_NaN());
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

		int n_cols =p_likelihood_2.size();
		double * objective    = new double[n_cols];//the objective coefficients
		double * col_lb       = new double[n_cols];//the column lower bounds
		double * col_ub       = new double[n_cols];//the column upper bounds

		for (int i = 0; i < n_cols; i++ ) 
		{
			objective[i]=p_likelihood_2[i];
			col_lb[i] = 0.0;
			col_ub[i] = si->getInfinity();
		}
		int N1=centroid_vec2.size();
		int N2=centroid_vec3.size();
		int n_rows = N1+N2;
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
			row_lb[i] = 0;
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

		vector<int> parent_2(N2, std::numeric_limits<double>::quiet_NaN());
		vector<vector<int>> t_id_2;
		get_matching_result(vsol,C_coeff_2,feature_vec2,feature_vec3,centroid_vec2,centroid_vec3,parent_2,t_id_2);

		step2_TT[i]=t_id_2;
		step2_parent_ids[i]=parent_2;

		
	}
	//end of void generate_structure ()
	 return 0;
}