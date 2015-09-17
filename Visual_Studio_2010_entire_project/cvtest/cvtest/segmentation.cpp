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


using namespace std;
using namespace cv;
using namespace flann;
using namespace alglib;
using namespace arma;
using namespace alglib_impl;
using namespace boost::geometry;
using boost::geometry::get;

const char* depthToStr(int depth) {
  switch(depth){
    case CV_8U: return "unsigned char";
    case CV_8S: return "char";
    case CV_16U: return "unsigned short";
    case CV_16S: return "short";
    case CV_32S: return "int";
    case CV_32F: return "float";
    case CV_64F: return "double";
  }
  return "invalid type!";
}

int bwmorph_majority(cv::Mat& curvedness, cv::Mat& Output)
{ 	
	cv::Mat Padded;
	copyMakeBorder(curvedness, Padded, 2,2,2,2, IPL_BORDER_REFLECT);
	int nRows = Padded.rows;
	int nCols = Padded.cols;
	int i,j;
   for( i = 0; i < nRows-4; i++)
	{
		double* rowPointer = Output.ptr<double>(i);
		for( j = 0; j < nCols-4; j++)
		{
			cv:Rect myROI(j,i,3,3);
			cv::Mat cropped = Padded(myROI);
			int count = cv::countNonZero(cropped);
			if (count>5)
				rowPointer[j]=1;
			else
				rowPointer[j]=0;
			//Output.at<double>(i,j)=scnd_highest;

		}
	}
	return 0;
}

double eccentricity2( vector<Point> &contour )
{	
	cv::Moments mu=moments(contour);
	double bigSqrt = sqrt( ( mu.m20 - mu.m02 ) *  ( mu.m20 - mu.m02 )  + 4 * mu.m11 * mu.m11  );
    return (double) ( mu.m20 + mu.m02 + bigSqrt ) / ( mu.m20 + mu.m02 - bigSqrt );
}

void labelBlobs(const cv::Mat &binary, vector<vector<cv::Point>> &blobs)
{
    blobs.clear();
 
    // Using labels from 2+ for each blob
    cv::Mat label_image(binary.size(), CV_64FC1);
	label_image.setTo(0);
	cv::Mat mask = (binary != 0);
	label_image.setTo(1, mask);
    vector<vector<double>> magvec;
	for(int j=0; j<label_image.rows; ++j)
		{
			vector<double> magvec_tmp;
			for(int k=0; k<label_image.cols; ++k)
			{
				magvec_tmp.push_back(label_image.at<double>(j,k));
			}
			magvec.push_back(magvec_tmp);
		}
    int label_count = 2; // starts at 2 because 0,1 are used already
 
    for(int y=0; y < binary.rows; y++) {
        for(int x=0; x < binary.cols; x++) {
            if(label_image.at<double>(y,x) == 0) {
                continue;
            }
			cv:: Mat label_image1;
			label_image.convertTo(label_image1,CV_8UC1);
            cv::Rect rect;
            cv::floodFill(label_image1, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);
			label_image1.convertTo(label_image,CV_64FC1);
 
            vector<cv::Point>  blob;
 
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(label_image.at<double>(i,j) != label_count) {
                        continue;
                    }
 
                    blob.push_back(cv::Point(j,i));
                }
            }
 
            blobs.push_back(blob);
 
            label_count++;
        }
    }
}


double vectorsum(vector<double> v)
{  double  total = 0;
   for (int i = 0; i < v.size(); i++)
      total += v[i];
   return total;
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
  result.push_back(last);
  return result;
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


std::vector<std::vector<int> > ocv_im2vec(cv::Mat img_C3) {
    vector<cv::Mat> channels(3);
	split(img_C3, channels);
	vector<vector<int>> im2vec;
		for(int i=0; i<3; ++i)
		{
			// get the channels (they follow BGR order in OpenCV)
			cv::Mat tempch=channels[i];
			vector<vector<int>> tmpchvec;
			vector<int> tmpchvec_psh;
			for(int j=0; j<tempch.rows; ++j)
			{
				vector<int> tmpchvec1;
				tempch.row(j).copyTo(tmpchvec1);
				tmpchvec.push_back(tmpchvec1);
			}
			tmpchvec=vec_transpose_4_int(tmpchvec);//Made it work as a col wise copy
			for(std::vector<std::vector<int> >::const_iterator it = tmpchvec.begin(), end = tmpchvec.end(); it != end; ++it)
			{
			  tmpchvec_psh.insert(tmpchvec_psh.end(), it->begin(), it->end());
			}
			im2vec.push_back(tmpchvec_psh);
		}
		return im2vec;
}

cv::Mat reshpe_back(const std::vector<std::vector<double> > channels_C, Size s){

	vector<cv::Mat> channels_revive;
		for (int k=0; k<3; ++k)
		{
			vector<vector<double>> intensity_ch;
			for (int i=0; i<s.width; ++i)
			{
				vector<double> intensity_t_tmp;
				for (int j=0; j<s.height; ++j)
				{
					intensity_t_tmp.push_back(channels_C[k][(i*1328)+j]);
				}
				intensity_ch.push_back(intensity_t_tmp);
			}
			intensity_ch=vec_transpose(intensity_ch);
			//create channel
			cv::Mat tmp_ch = cv::Mat::zeros(s.height, s.width, CV_8UC1);
			for(int i=0; i<tmp_ch.rows; ++i)
			{
			 for(int j=0; j<tmp_ch.cols; ++j)
			 {
				  //cout<<int(intensity_ch.at(i).at(j))<<endl;
				  tmp_ch.at<unsigned char>(i, j) = unsigned char(int(round(intensity_ch.at(i).at(j))));
				  //tmp_ch.at<int>(i, j) =  int(round(intensity_ch.at(i).at(j)));
			 }
			}

			channels_revive.push_back(tmp_ch);
		}
		cv::Mat intensity;
		merge(channels_revive, intensity);
		return intensity;

}

std::vector<std::vector<double> > vec_deconvolution_normalize(const std::vector<std::vector<int> > im2vec) {

	vector<vector<double>> y_OD;
	double eps=std::numeric_limits<double>::epsilon();
	for(int i=0; i<im2vec.size(); ++i)
	{
		vector<double> temp_y_OD;
		for(int j=0; j<im2vec[i].size(); ++j)
		{
			double temp=double(im2vec[i][j])+eps;
			temp=temp/255;
			temp=log(temp);
			temp_y_OD.push_back(-temp);
		}
		y_OD.push_back(temp_y_OD);
	}
	return y_OD;
}

std::vector<std::vector<double> > vec_deconvolution_denormalize(const std::vector<std::vector<double> > im2vec) {

	vector<vector<double>> y_OD;
	double eps=std::numeric_limits<double>::epsilon();
	for(int i=0; i<im2vec.size(); ++i)
	{
		vector<double> temp_y_OD;
		for(int j=0; j<im2vec[i].size(); ++j)
		{
			double temp=double(im2vec[i][j])+eps;
			temp=exp(-temp);
			temp=temp*255;
			temp_y_OD.push_back(temp);
		}
		y_OD.push_back(temp_y_OD);
	}
	return y_OD;
}


cv::Mat Ocv_ColorDeconvolution(const std::vector<std::vector<double> > stains, cv::Mat img, Size s)
{
	//calculate stain intensities using color deconvolution
		//ColorDeconvolution(img, stains, [true true true])

		//normalize stains
		vector<vector<double>> stain_t=vec_transpose(stains);

		for (int i = 0; i < stain_t.size(); i++ ) {
			if(round(sqrt(norm(stain_t[i])))==1)
			{
				std::transform(stain_t[i].begin(), stain_t[i].end(), stain_t[i].begin(), std::bind1st(std::multiplies<double>(),(1/round(sqrt(norm(stain_t[i]))))));
			}
		}

		vector<vector<double>> stain_norm=vec_transpose(stain_t);

		/*
		//Alglib matrix inverse
		double temp[3][3];
		for (int i = 0; i < stain_norm.size(); i++ ){
			for (int j = 0; j < stain_norm[i].size(); j++ ){
				temp[i][j]=stain_norm[i][j];
				cout<<temp[i][j]<<endl;
			}
		}
		alglib::real_2d_array matrix;
		matrix.setcontent(3, 3, *temp);
		alglib::ae_int_t info;
		alglib::matinvreport rep;
		rmatrixinverse(matrix, info, rep);
		cout<<int(info)<<endl;
		printf("%s\n", matrix.tostring(9).c_str());
		*/
		//OpenCV matrix inverse
		cv::Mat Q= cv::Mat(stain_norm.size(), stain_norm.at(0).size(), CV_64FC1);
		for(int i=0; i<Q.rows; ++i)
		{
		 for(int j=0; j<Q.cols; ++j)
		 {
			  Q.at<double>(i, j) = double(stain_norm.at(i).at(j));
		 }
		}
		cout<<Q<<endl;
		Q=Q.inv();
		cout<<Q<<endl;

		//ocv_im2vec(I) (converts color image to 3 x MN matrix)
		cv::Mat img_C3( 3, 4, CV_8SC3, CV_RGB(3,2,1) );
		vector<vector<int>> im2vec=ocv_im2vec(img);// channel follow BGR order in OpenCV in Matlab RGB
		//deconvolution_normalize
		vector<vector<double>> y_OD1=vec_deconvolution_normalize(im2vec);
		//Switch B & R channel for multiplication
		/*
		 vector<double> tmP_switch=y_OD1[2];
		 y_OD1[2]=y_OD1[0];
		 y_OD1[0]=tmP_switch;
		 */
		cv::Mat y_OD= cv::Mat(y_OD1.size(), y_OD1.at(0).size(), CV_64FC1);
		for(int i=0; i<y_OD.rows; ++i)
		{
		 for(int j=0; j<y_OD.cols; ++j)
		 {
			  y_OD.at<double>(i, j) = y_OD1.at(i).at(j);
		 }
		}
		cv::Mat C_mat(y_OD.size(),CV_64FC1);
		C_mat=Q*y_OD;
		vector<vector<double>> C;
		for(int i=0; i<C_mat.rows; ++i)
		{
			vector<double> C_tmp;
		 for(int j=0; j<C_mat.cols; ++j)
		 {
			  C_tmp.push_back(C_mat.at<double>(i, j));
		 }
		 C.push_back(C_tmp);
		}
		//Switch back B & R channel after multiplication
		/*
		 vector<double> tmP_switch1=C[2];
		 C[2]=C[0];
		 C[0]=tmP_switch1;
		 */
		 //deconvolution_denormalize
		vector<vector<double>> channels_C=vec_deconvolution_denormalize(C);


		cv::Mat intensity=reshpe_back(channels_C,s);

		

		//reshape back to an image(Inside function)
		/*
		vector<cv::Mat> channels_revive;
		for (int k=0; k<3; ++k)
		{
			vector<vector<double>> intensity_ch;
			for (int i=0; i<s.width; ++i)
			{
				vector<double> intensity_t_tmp;
				for (int j=0; j<s.height; ++j)
				{
					intensity_t_tmp.push_back(channels_C[k][(i*1328)+j]);
				}
				intensity_ch.push_back(intensity_t_tmp);
			}
			intensity_ch=vec_transpose(intensity_ch);
			//create channel
			cv::Mat tmp_ch = cv::Mat::zeros(s.height, s.width, CV_8UC1);
			for(int i=0; i<tmp_ch.rows; ++i)
			{
			 for(int j=0; j<tmp_ch.cols; ++j)
			 {
				  //cout<<int(intensity_ch.at(i).at(j))<<endl;
				  tmp_ch.at<unsigned char>(i, j) = unsigned char(int(round(intensity_ch.at(i).at(j))));
			 }
			}

			channels_revive.push_back(tmp_ch);
		}
		cv::Mat intensity;
		merge(channels_revive, intensity);
		*/

		
		//generate color image associated with individual stains
		vector<cv::Mat> colorStainImages;
		for (int i=0; i<3; ++i)
		{
			cv::Mat M= cv::Mat(stain_t[i].size(),1, CV_64FC1);
			for(int j=0; j<M.rows; ++j)
			{
				M.at<double>(j,0) =double(stain_t[i][j]);
			}
			cv::Mat Ci= cv::Mat(1,C[i].size(), CV_64FC1);
			for(int j=0; j<Ci.cols; ++j)
			{
				Ci.at<double>(0,j)=double(C[i][j]);
			}
			cv::Mat stain_OD1(M.rows,Ci.cols,CV_64FC1);
			stain_OD1=M*Ci;
			vector<vector<double>> stain_RGB_vec;
			for(int j=0; j<stain_OD1.rows; ++j)
			{
				vector<double> stain_RGB_tmp;
				for(int k=0; k<stain_OD1.cols; ++k)
				{
					stain_RGB_tmp.push_back(stain_OD1.at<double>(j,k));
				}
				stain_RGB_vec.push_back(stain_RGB_tmp);
			}
			vector<vector<double>> stain_RGB_vec1=vec_deconvolution_denormalize(stain_RGB_vec);
			/*
			cout<<stain_RGB_vec1[0][554433]<<endl;
			cout<<stain_RGB_vec1[1][554433]<<endl;
			cout<<stain_RGB_vec1[2][554433]<<endl;
			cout<<stain_RGB_vec1[0][554435]<<endl;
			cout<<stain_RGB_vec1[1][554435]<<endl;
			cout<<stain_RGB_vec1[2][554435]<<endl;
			*/
			cv::Mat tmp1=reshpe_back(stain_RGB_vec1,s);
			colorStainImages.push_back(tmp1);
			/*
			Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
			cv::Mat img1;
			resize(tmp1, img1, img_size);//50% redeuction in display to fit in display-view
			imshow("intensity_img", img1);
			waitKey(5);
			*/
		}
		//ColorDeconvolution(img, stains, [true true true]) Ends
		return intensity;
}

std::vector<cv::Mat > ocv_Hessian2D(cv::Mat DAB, double sigmas){

	vector<cv::Mat> Hessian2D;
	//Hessian2D(I[DAB],Sigma)
	vector<double> lingrid=linspace_intrvl(-round(3*sigmas),round(3*sigmas),1);
	//Make kernel coordinates
	vector<vector<double>> X_ndgrid;
	vector<vector<double>> Y_ndgrid;
	for (int j=0; j<lingrid.size(); ++j)
	{
		Y_ndgrid.push_back(lingrid);
	}
	X_ndgrid=vec_transpose(Y_ndgrid);
	cv::Mat X_mat= cv::Mat(X_ndgrid.size(), X_ndgrid.at(0).size(), CV_64FC1);
	cv::Mat Y_mat= cv::Mat(Y_ndgrid.size(), Y_ndgrid.at(0).size(), CV_64FC1);
	for(int k=0; k<X_mat.rows; ++k)
	{
		for(int j=0; j<X_mat.cols; ++j)
		{
			X_mat.at<double>(k, j) = double(X_ndgrid.at(k).at(j));
			Y_mat.at<double>(k, j) = double(Y_ndgrid.at(k).at(j));
		}
	}
	cout<<X_mat<<endl;
	cout<<Y_mat<<endl;
	const double pi = boost::math::constants::pi<double>();
	//Build the gaussian 2nd derivatives filters
	cv::Mat DGaussx_trm2=-1*(((X_mat.mul(X_mat)) + (Y_mat.mul(Y_mat)))/(2*pow(sigmas,2)));
	cv::exp(DGaussx_trm2,DGaussx_trm2);
	cv::Mat DGaussxx= 1/(2*pi*pow(sigmas,4)) * (((X_mat.mul(X_mat))/pow(sigmas,2)) - 1) .mul( DGaussx_trm2);
	cv::Mat DGaussxy= 1/(2*pi*pow(sigmas,6)) * (X_mat.mul(Y_mat)) .mul( DGaussx_trm2);
	/*
	vector<vector<double>> DGaussxxvec;
	for(int j=0; j<DGaussxx.rows; ++j)
	{
		vector<double> DGaussxxvec_tmp;
		for(int k=0; k<DGaussxx.cols; ++k)
		{
			DGaussxxvec_tmp.push_back(DGaussxx.at<double>(j,k));
		}
		DGaussxxvec.push_back(DGaussxxvec_tmp);
	}

	vector<vector<double>> DGaussxyvec;
	for(int j=0; j<DGaussxy.rows; ++j)
	{
		vector<double> DGaussxyvec_tmp;
		for(int k=0; k<DGaussxy.cols; ++k)
		{
			DGaussxyvec_tmp.push_back(DGaussxy.at<double>(j,k));
		}
		DGaussxyvec.push_back(DGaussxyvec_tmp);
	}
	*/

	cv::Mat DGaussyy;
	cv::transpose(DGaussxx,DGaussyy);
	cv::Mat Dxx1,Dxy1,Dyy1;
	Point anchor( 0 ,0 );
	double delta = 0;
	cv::Mat DABpadded;
	//Number of padding {[size(lingrid)-1]/2}
	copyMakeBorder(DAB,DABpadded,1,1,1,1,BORDER_CONSTANT,Scalar(0));
	for(int j=0; j<((lingrid.size()-1)/2)-1; ++j)
	{
	copyMakeBorder(DABpadded,DABpadded,1,1,1,1,BORDER_CONSTANT,Scalar(0));
	}
	// functionality: N-D filtering of multidimensional images

	filter2D(DABpadded, Dxx1, CV_64FC1 , DGaussxx,anchor);
	filter2D(DABpadded, Dxy1, CV_64FC1 , DGaussxy,anchor);
	filter2D(DABpadded, Dyy1, CV_64FC1 , DGaussyy,anchor);
	



	cv::Mat Dxx(DAB.rows, DAB.cols, CV_64FC1);
	cv::Mat Dxy(DAB.rows, DAB.cols, CV_64FC1);
	cv::Mat Dyy(DAB.rows, DAB.cols, CV_64FC1);



	Dxx=Dxx1(Rect(0, 0, DAB.cols,DAB.rows));
	Dxy=Dxy1(Rect(0, 0, DAB.cols,DAB.rows));
	Dyy=Dyy1(Rect(0, 0, DAB.cols,DAB.rows));


	std::cout<<depthToStr(Dxx.depth())<< std::endl;
	std::cout<<depthToStr(Dxy.depth())<< std::endl;
	std::cout<<depthToStr(Dyy.depth())<< std::endl;
	/*
	vector<vector<double>> Dxxvec;
	for(int j=0; j<Dxx.rows; ++j)
		{
		vector<double> DGauss0vec_tmp;
		for(int k=0; k<Dxx.cols; ++k)
		{
			DGauss0vec_tmp.push_back(Dxx.at<double>(j,k));
		}
		Dxxvec.push_back(DGauss0vec_tmp);
		}

	vector<vector<double>> Dxyvec;
	for(int j=0; j<Dxy.rows; ++j)
		{
		vector<double> DGauss1vec_tmp;
		for(int k=0; k<Dxy.cols; ++k)
		{
			DGauss1vec_tmp.push_back(Dxy.at<double>(j,k));
		}
		Dxyvec.push_back(DGauss1vec_tmp);
		}
	
	vector<vector<double>> Dyyvec;
	for(int j=0; j<Dyy.rows; ++j)
		{
		vector<double> DGauss2vec_tmp;
		for(int k=0; k<Dyy.cols; ++k)
		{
			DGauss2vec_tmp.push_back(Dyy.at<double>(j,k));
		}
		Dyyvec.push_back(DGauss2vec_tmp);
		}
	*/
	//Dxx.convertTo(Dxx, CV_64FC1);
	//Dxy.convertTo(Dxy, CV_64FC1);
	//Dyy.convertTo(Dyy, CV_64FC1);
	
	Hessian2D.push_back(Dxx);
	Hessian2D.push_back(Dxy);
	Hessian2D.push_back(Dyy);

	return Hessian2D;

}

 
cv::Mat ocv_multiScaleFilter2D(cv::Mat DAB, Size s)
{
    //Set options value
	vector <double> options_ScaleRange;
	double myints[] = {1,5};
	options_ScaleRange.assign(myints,myints+3);
	double options_ScaleRatio=0.2;
	double  options_BetaOne=0.5;
	double  options_BetaTwo=15;

	vector<double> sigmas=linspace_intrvl(options_ScaleRange.at(0),options_ScaleRange.at(1),options_ScaleRatio);
	double Beta=2*pow(options_BetaOne,2);
	double c=2*pow(options_BetaTwo,2);
	//**container vector<cv::Mat> ALLfiltered (each mat:Size of image 1328X1366, 21 mat size of sigma)
	//**container vector<cv::Mat> ALLangles (each mat:Size of image 1328X1366, 21 mat size of sigma)
	//**vector<vector<cv::Mat>>  
	cv::Mat outIm;
	cv::Mat whatScale(s, CV_32SC1);
	cv::Mat mask_comp;
	cv::Mat Direction;
	for (int i=0; i<sigmas.size(); ++i)
	{
			cout<<"Current Filter Sigma: "<< sigmas[i]<<endl;

			vector<cv::Mat> DGauss=ocv_Hessian2D( DAB, sigmas[i]);			 

		/*
	        Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
			cv::Mat img1;
			resize(DGauss[0], img1, img_size);//50% redeuction in display to fit in display-view
			imshow("Dxx", img1);
			waitKey(5);
			cv::Mat img2;
			resize(DGauss[1], img2, img_size);//50% redeuction in display to fit in display-view
			imshow("Dxy", img2);
			waitKey(5);
			cv::Mat img3;
			resize(DGauss[2], img3, img_size);//50% redeuction in display to fit in display-view
			imshow("Dyy", img3);
			waitKey(0);
			*/
		

			// Correct for scale
			DGauss[0]=DGauss[0].mul(pow(sigmas[i],2));
			DGauss[1]=DGauss[1].mul(pow(sigmas[i],2));
			DGauss[2]=DGauss[2].mul(pow(sigmas[i],2));
			//std::cout<<depthToStr(DGauss[0].depth())<< std::endl;
			//std::cout<<depthToStr(DGauss[1].depth())<< std::endl;
			//std::cout<<depthToStr(DGauss[2].depth())<< std::endl;

			//[Lambda1,Lambda2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)

			cv::Mat tmp1(DGauss[0].size(),DGauss[0].type()),tmp(DGauss[0].size(),DGauss[0].type());
			tmp1=(DGauss[0]-DGauss[2]).mul(DGauss[0]-DGauss[2])+4*(DGauss[1].mul(DGauss[1]));
			cv::sqrt(tmp1,tmp);
			/*
			vector<vector<double>> tmpvec;
			for(int j=0; j<tmp.rows; ++j)
			{
			vector<double> tmpvec_tmp;
			for(int k=0; k<tmp.cols; ++k)
			{
				tmpvec_tmp.push_back(tmp.at<double>(j,k));
			}
			tmpvec.push_back(tmpvec_tmp);
			}
			*/
			cv::Mat v2x1;
			v2x1=2*DGauss[1];
			cv::Mat v2y1;
			v2y1=DGauss[2]-DGauss[0]+tmp;
			cv::Mat mag1,mag;
			mag1=v2x1.mul(v2x1)+v2y1.mul(v2y1);
			cv::sqrt(mag1,mag);
			/*
			vector<vector<double>> magvec;
			for(int j=0; j<mag.rows; ++j)
			{
			vector<double> magvec_tmp;
			for(int k=0; k<mag.cols; ++k)
			{
				magvec_tmp.push_back(mag.at<double>(j,k));
			}
			magvec.push_back(magvec_tmp);
			}
			*/
			std::cout<<depthToStr(mag.depth())<< std::endl;
			cv::Mat mask = (mag == 0);
			mag.setTo(1, mask);
			std::cout<<depthToStr(mag.depth())<< std::endl;
			cv::Mat v2x,v2y;
			v2x=v2x1/mag;
			v2y=v2y1/mag;
			cv::Mat v1x,v1y;
			v1x=-1*v2y;
			v1y=v2x;
			cv::Mat mu1,mu2;
			mu1 = 0.5*(DGauss[0] + DGauss[2] + tmp);
			mu2 = 0.5*(DGauss[0] + DGauss[2] - tmp);
			cv::Mat check = (cv::abs(mu1)>cv::abs(mu2));
			cv::Mat Lambda1=mu1;mu2.copyTo(Lambda1,check);
			cv::Mat Lambda2=mu2;mu1.copyTo(Lambda2,check);
			/*
			vector<vector<double>> Lambda1vec;
			for(int j=0; j<Lambda1.rows; ++j)
			{
			vector<double> Lambda1_tmp;
			for(int k=0; k<Lambda1.cols; ++k)
			{
				Lambda1_tmp.push_back(Lambda1.at<double>(j,k));
			}
			Lambda1vec.push_back(Lambda1_tmp);
			}

			vector<vector<double>> Lambda2vec;
			for(int j=0; j<Lambda2.rows; ++j)
			{
			vector<double> Lambda2_tmp;
			for(int k=0; k<Lambda2.cols; ++k)
			{
				Lambda2_tmp.push_back(Lambda2.at<double>(j,k));
			}
			Lambda2vec.push_back(Lambda2_tmp);
			}
			*/
			cv::Mat Ix=v1x; v2x.copyTo(Ix,check);
			cv::Mat Iy=v1y; v2y.copyTo(Iy,check);

			vector<vector<double>> Ixvec;
			for(int j=0; j<Ix.rows; ++j)
			{
			vector<double> Ix_tmp;
			for(int k=0; k<Ix.cols; ++k)
			{
				Ix_tmp.push_back(Ix.at<double>(j,k));
			}
			Ixvec.push_back(Ix_tmp);
			}

			vector<vector<double>> Iyvec;
			for(int j=0; j<Iy.rows; ++j)
			{
			vector<double> Iy_tmp;
			for(int k=0; k<Iy.cols; ++k)
			{
				Iy_tmp.push_back(Iy.at<double>(j,k)); 
			}
			Iyvec.push_back(Iy_tmp);
			}
			//[Lambda1,Lambda2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy) ends
			vector<vector<double>> angles_vec;

			for(int j=0; j<Ixvec.size(); ++j)
			{
				vector<double> angles_vec_tmp;
				for(int k=0; k<Ixvec[0].size(); ++k)
				{
					angles_vec_tmp.push_back(atan2(Iyvec[j][k],Ixvec[j][k]));
				}
				angles_vec.push_back(angles_vec_tmp);
			}

			cv::Mat angles= cv::Mat(angles_vec.size(), angles_vec.at(0).size(), CV_64FC1);
			for(int k=0; k<angles.rows; ++k)
			{
			for(int j=0; j<angles.cols; ++j)
			{
				angles.at<double>(k, j) = double(angles_vec.at(k).at(j));
			}
			}
			/*
			cv::Mat img4;
			Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
			resize(angles, img4, img_size);
			imshow("Angles", img4);
			waitKey(0);	
			*/
			double eps=std::numeric_limits<double>::epsilon();
			cv::Mat mask1 = (Lambda2 == 0);
			Lambda2.setTo(eps, mask1);
			cv::Mat Rb1,Rb;
			Rb1=Lambda1/Lambda2;
			Rb=Rb1.mul(Rb1);

			cv::Mat S2;
			S2=Lambda1.mul(Lambda1)+Lambda2.mul(Lambda2);

			//case 'axons'
			cv::Mat Rbprs;
			Rbprs=-1*Rb;
			Rbprs=Rbprs.mul(1/Beta);
			cv::exp(Rbprs,Rbprs);

			cv::Mat S2prs;
			S2prs=-1*S2; 
			S2prs=S2prs.mul(1/c);
			cv::exp(S2prs,S2prs);

			cv::Mat One_mat(DGauss[0].size(), DGauss[0].type());
			One_mat.setTo(1);

			cv::Mat Itmp;
			Itmp=One_mat-S2prs;
			cv::Mat Ifiltered=Rbprs.mul(Itmp);

			Ifiltered=30*Ifiltered;//debug

			cv::Mat mask2 = (Lambda2 < 0);
			Ifiltered.setTo(0, mask2);

			//debug
			cv::Mat mask3 = (Ifiltered == 0);
			//std::cout<<depthToStr(mask3.depth())<< std::endl;
			
			//debug

			vector<vector<double>> Ifilteredvec;
			vector<double> sumifltr1;
			vector<double> summsk;
			for(int j=0; j<Ifiltered.rows; ++j)
			{
			vector<double> Ifiltered_tmp;
			vector<double> msksm_tmp;
			for(int k=0; k<Ifiltered.cols; ++k)
			{
				Ifiltered_tmp.push_back(Ifiltered.at<double>(j,k));
				msksm_tmp.push_back(double(mask3.at<unsigned char>(j,k)));
			}
			Ifilteredvec.push_back(Ifiltered_tmp);
			sumifltr1.push_back(vectorsum(Ifiltered_tmp));
			summsk.push_back(vectorsum(msksm_tmp));
			}
			//cout<<vectorsum(sumifltr1)<<endl;
			//cout<<vectorsum(summsk)<<endl;
			//cout<<vectorsum(summsk)/255<<endl;

			/*
			cv::Mat img3;
			//Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
			resize(Ifiltered, img3, img_size);
			imshow("Ifiltered", img3);
			waitKey(0);
			*/

			//Saving the max value & the index of scale/ sigma value
			 
			if (i==0)
			{
				outIm=Ifiltered;		
				Direction=angles;
				whatScale.setTo(0);
			}
			else
			{
				mask_comp = (Ifiltered>outIm);
				std::cout<<depthToStr(Ifiltered.depth())<< std::endl;
				std::cout<<depthToStr(outIm.depth())<< std::endl;
				outIm.copyTo(Ifiltered,mask_comp);
				whatScale.setTo(i,mask_comp);
				Direction.copyTo(angles,mask_comp);
			}


	}
		
	//multiScaleFilter2D(I, type, options) Ends
	return outIm;
}



int _tmain(int argc, _TCHAR* argv[])
{
	 CFileFind finder;
	 string img_dirPath;
	 img_dirPath="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/data_to_segment/";
	 //Reading number of files in the directory
	 int num_img=0;
	 DIR *dir;
	 struct dirent *ent;
	 
	 if ((dir = opendir ("C:\\Users\\Sayan\\Dropbox\\GSoC_2015_Implementation\\data_to_segment\\")) != NULL) {
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
	 dirPathC = dirPathC1 + "\\*.bmp";
	 BOOL bWorking = finder.FindFile(dirPathC);
	 while (bWorking)

	{
		 //Read the registered image
		CString dirPathC2;
		dirPathC2="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/data_to_segment/";
		bWorking = finder.FindNextFile();
		cstr = finder.GetFileName();
		CStringA cstr1(cstr);
		CString finpath=dirPathC1+cstr;
		_tprintf_s(_T("%s\n"), (LPCTSTR) finpath);
		_tprintf_s(_T("%s\n"), (LPCTSTR) cstr);
		dirPathC2=dirPathC2+cstr;
	    CT2CA pszConvertedAnsiString (dirPathC2);
	    string image_name1(pszConvertedAnsiString);
		char *imagename=&image_name1[0];
		cv::Mat imgGray;//Initialize a matrix to store the gray version of color image
		cv::Mat img = imread(imagename); //Load the registered Image
		cout<<depthToStr(img.depth())<<endl;
		Size s = img.size();
		/*
		//Display the 50% Reduction of the loaded image
		cout<<"Image Height:"<<s.height<<endl;
		cout<<"Image Width:"<<s.width<<endl;
		Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
		cv::Mat img1;
		resize(img, img1, img_size);//50% redeuction in display to fit in display-view
		imshow("Registered_Image", img1);	
		waitKey(5000);
		//Display Until Here
		*/
		//process lumens
		double G = 225;
		double eccThr = 1;
		vector <double> lumenAreaThr;
		double myints[] = {50,150000};
		lumenAreaThr.assign(myints,myints+2);
		vector<cv::Mat> imgch(3);
		split(img, imgch);
		imgch[0].convertTo(imgch[0], CV_64FC1);
		cv::Mat mask1 = (imgch[0] >G);
		imgch[1].convertTo(imgch[1], CV_64FC1);
		cv::Mat mask2 = (imgch[1] >G);
		imgch[2].convertTo(imgch[2], CV_64FC1);
		cv::Mat mask3 = (imgch[2] >G);
		cv::Mat mask2inv = (mask2==0);
		cv::Mat mask3inv = (mask3==0);
		mask1.setTo(0,mask2inv);
		mask1.setTo(0,mask3inv);

		cv::Mat bwLumenMask(s,CV_64FC1);
		bwLumenMask.setTo(0);
		bwLumenMask.setTo(1,mask1);

		Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
		cv::Mat img1;
		resize(bwLumenMask, img1, img_size);//50% redeuction in display to fit in display-view
		imshow("bwLumenMask", img1);
		waitKey(5);

		  

		//imclearborder

		//Matlab
		//a matrix of same size with 1
		 
		cv::Mat im2(bwLumenMask.size(),CV_64FC1);
		im2.setTo(1);
		/*
		vector<vector<double>> im2vec;
		for(int j=0; j<im2.rows; ++j)
		{
			vector<double> im2vec_tmp;
			for(int k=0; k<im2.cols; ++k)
			{
				im2vec_tmp.push_back(im2.at<double>(j,k));
			}
			im2vec.push_back(im2vec_tmp);
		}
		*/
		//copy make border with 0
		copyMakeBorder(im2,im2,1,1,1,1,BORDER_CONSTANT,Scalar(0));
		/*
		vector<vector<double>> im2vecpad;
		for(int j=0; j<im2.rows; ++j)
		{
			vector<double> im2vec_tmp1;
			for(int k=0; k<im2.cols; ++k)
			{
				im2vec_tmp1.push_back(im2.at<double>(j,k));
			}
			im2vecpad.push_back(im2vec_tmp1);
		}
		*/		
		// erode with 8 connection 
		cv::Mat im21;
		erode(im2,im21,getStructuringElement(MORPH_RECT, Size (3,3)));
		
		//make a idx vector 
		cv::Mat im22(bwLumenMask.size(),CV_64FC1);
		im21(Rect(1, 1, im21.cols-2,im21.rows-2)).copyTo(im22);
		/*
		vector<vector<double>> im22vec;
		for(int j=0; j<im22.rows; ++j)
		{
			vector<double> im22vec_tmp;
			for(int k=0; k<im22.cols; ++k)
			{
				im22vec_tmp.push_back(im22.at<double>(j,k));
			}
			im22vec.push_back(im22vec_tmp);
		}
		*/
		//Morphological reconstruction


		cv::Mat imgBWcopy =bwLumenMask;
		vector<Vec4i> hierarchy;
		vector<vector<Point>> contours;
		cv::Mat img225;
		imgBWcopy.convertTo(img225,CV_8U);
		findContours(img225,contours,hierarchy,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
		int imgRows=imgBWcopy.rows;int imgCols=imgBWcopy.cols;int radius1=650;int radius2=650;
		vector<int> contourList; // ID list of contours that touch the border
		// For each contour...
		for (int i = 0; i < contours.size(); i++ ) 
		{
			//Get the i'th contour
			vector<Point> cnt=contours[i];
			// Look at each point in the contour
			for (int j = 0; j < cnt.size(); j++) 
			{
				bool check1,check2;
				check1=(cnt[j].x>=0 & cnt[j].x<radius1)|(cnt[j].x>=imgRows-1-radius1 & cnt[j].x<imgRows);
				check2=(cnt[j].y>=0 & cnt[j].y<radius2)|(cnt[j].y>=imgRows-1-radius2 & cnt[j].y<imgRows);

				if (check1|check2)
				{
					contourList.push_back(i);
					break;
				}
			}
		}
		cv::Mat imclrbrdr1(bwLumenMask.size(),CV_64FC1);
		imclrbrdr1.setTo(1);
		for (int i = 0; i < contourList.size(); i++ ) 
		{
			//drawContours(imgBWcopy,contours,i,Scalar(0),CV_FILLED);
			drawContours(imclrbrdr1,contours,i,Scalar(0),CV_FILLED);
		}
		
		cv::Mat img21;
		resize(imclrbrdr1, img21, img_size);//50% redeuction in display to fit in display-view
		imshow("imclrbrdr1", img21);
		waitKey(5);
		/*
		cv::Mat img2;
		resize(imgBWcopy, img2, img_size);//50% redeuction in display to fit in display-view
		imshow("imgBWcopy", img2);
		waitKey(5);
		*/
		cout<<depthToStr(bwLumenMask.depth())<<endl;
		cout<<depthToStr(imgBWcopy.depth())<<endl;

		cv::Mat imclrbrdr(bwLumenMask.size(),CV_64FC1);
		imclrbrdr= bwLumenMask - imclrbrdr1;

		cout<<depthToStr(imclrbrdr.depth())<<endl;



		/*
		vector<vector<double>> bwLumenMaskvec;
		vector<double> sum1;
		for(int j=0; j<bwLumenMask.rows; ++j)
		{
			vector<double>  bwLumenMask_tmp;
			for(int k=0; k<bwLumenMask.cols; ++k)
			{
				bwLumenMask_tmp.push_back(bwLumenMask.at<double>(j,k));
			}
			bwLumenMaskvec.push_back(bwLumenMask_tmp);
			sum1.push_back(vectorsum(bwLumenMask_tmp));
		}
		cout<<vectorsum(sum1)<<endl;

		vector<double> sum2;
		vector<vector<double>> imclrbrdr1vec;
		for(int j=0; j<imclrbrdr1.rows; ++j)
		{
			vector<double>  imclrbrdr1_tmp;
			for(int k=0; k<imclrbrdr1.cols; ++k)
			{
				imclrbrdr1_tmp.push_back(imclrbrdr1.at<double>(j,k));
			}
			imclrbrdr1vec.push_back(imclrbrdr1_tmp);
			sum2.push_back(vectorsum(imclrbrdr1_tmp));
		}
		cout<<vectorsum(sum2)<<endl;

		vector<double> sum3;
		vector<vector<double>> imclrbrdrvec;
		for(int j=0; j<imclrbrdr.rows; ++j)
		{
			vector<double>  imclrbrdr_tmp;
			for(int k=0; k<imclrbrdr.cols; ++k)
			{
				imclrbrdr_tmp.push_back(imclrbrdr.at<double>(j,k));
			}
			imclrbrdrvec.push_back(imclrbrdr_tmp);
			sum3.push_back(vectorsum(imclrbrdr_tmp));
		}
		cout<<vectorsum(sum3)<<endl;
		*/
		cv::Mat img3;
		resize(imclrbrdr, img3, img_size);//50% redeuction in display to fit in display-view
		imshow("imclrbrdr", img3);
		waitKey(5);
		cv::Mat imclrbrdr2(bwLumenMask.size(),CV_64FC1);
		bwmorph_majority(imclrbrdr,imclrbrdr2);
		cv::Mat img4;
		resize(imclrbrdr2, img4, img_size);//50% redeuction in display to fit in display-view
		imshow("imclrbrdr2", img4);
		waitKey(5);
		vector<Vec4i> hierarchy1;
		vector<vector<Point>> contours1;
		cv::Mat img226;
		imclrbrdr2.convertTo(img226,CV_8U);
		findContours(img226,contours1,hierarchy1,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
		cv::Mat imclrbrdr3(bwLumenMask.size(),CV_64FC1);
		imclrbrdr3.setTo(0);

		for (int i = 0; i < contours1.size(); i++ ) 
		{
			if ((contours1[i].size()>3)&(contourArea(contours1[i])>30))//add (contourArea(contours1[i])<150000)
				drawContours(imclrbrdr3,contours1,i,Scalar(1),CV_FILLED);
		}

		//bwlabel matlab try
		//vector<vector<cv::Point>> labelled_blobs;
		//labelBlobs(imclrbrdr3,  labelled_blobs);

		cv::Mat img5;
		resize(imclrbrdr3, img5, img_size);//50% redeuction in display to fit in display-view
		imshow("imclrbrdr2_fill", img5);	
		waitKey(5);

		cv::Mat imclrbrdr4;
		morphologyEx(imclrbrdr3,imclrbrdr4,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE , Size (5,5)));

		cv::Mat img6;
		resize(imclrbrdr4, img6, img_size);//50% redeuction in display to fit in display-view		
		imshow("imclrbrdr4", img6);	
		waitKey(5);
		
		cv::Mat imclrbrdr5;
		dilate(imclrbrdr4,imclrbrdr5,getStructuringElement(MORPH_ELLIPSE , Size (9,9)));

		cv::Mat img7;
		resize(imclrbrdr5, img7, img_size);//50% redeuction in display to fit in display-view		
		imshow("imclrbrdr5", img7);	
		waitKey(5);

		cv::Mat ringBW(bwLumenMask.size(),CV_64FC1);
		ringBW=imclrbrdr5-imclrbrdr4;

		cv::Mat img8;
		resize(ringBW, img8, img_size);//50% redeuction in display to fit in display-view		
		imshow("ringBW", img8);	
		waitKey(5);//***Save ringBW***

		CT2CA pszConvertedAnsiString1 (cstr);
	    string image_name11(pszConvertedAnsiString1);
		char *imagename2=&image_name11[0];
		char *extn=".phi0.yml"; 
		string file1 = string(imagename2)+string(extn);
		const char *file2=file1.c_str();
		cv::FileStorage storage(file2, cv::FileStorage::WRITE);
		storage << "ringBW" << ringBW;
		storage.release();

		/*
		//Applying Canny
		cv::Mat canny_output,canny_ip;
		imclrbrdr3.convertTo(canny_ip,CV_8UC1);
		Canny( canny_ip, canny_output, 1, 1, 3 );
		
		*/

		/*
		cv::Mat imgBWcopy =bwLumenMask;
		vector<Vec4i> hierarchy;
		vector<vector<Point>> contours;
		cv::Mat img225;
		imgBWcopy.convertTo(img225,CV_8U);
		findContours(img225,contours,hierarchy,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
		int imgRows=imgBWcopy.rows;int imgCols=imgBWcopy.cols;int radius=300;
		vector<int> contourList; // ID list of contours that touch the border
		// For each contour...
		for (int i = 0; i < contours.size(); i++ ) 
		{
			//Get the i'th contour
			vector<Point> cnt=contours[i];
			// Look at each point in the contour
			for (int j = 0; j < cnt.size(); j++) 
			{
				bool check1,check2;
				check1=(cnt[j].x>=0 & cnt[j].x<radius)|(cnt[j].x>=imgRows-1-radius & cnt[j].x<imgRows);
				check2=(cnt[j].y>=0 & cnt[j].y<radius)|(cnt[j].y>=imgRows-1-radius & cnt[j].y<imgRows);

				if (check1|check2)
				{
					contourList.push_back(i);
					break;
				}
			}
		}

		for (int i = 0; i < contourList.size(); i++ ) 
		{
			drawContours(imgBWcopy,contours,i,Scalar(0, 0, 0),CV_FILLED);
		}

		
		//Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
		cv::Mat img2;
		resize(imgBWcopy, img2, img_size);//50% redeuction in display to fit in display-view
		imshow("imgBWcopy", img2);
		waitKey(0);
		*/


		//define OD matrix; each column is associated with one stain (i.e. Hematoxylin, DAB, and red_marker)
		vector<vector<double>> stains;
		
		double blue[3] = {0.286, 0.731, 0.711};
		vector<double> Blue(&blue[0], &blue[0]+3);
		double green[3] = {0.704, 0.570, 0.696};
		vector<double> Green(&green[0], &green[0]+3);
		double red[3] = {0.650, 0.368, 0.103};
		vector<double> Red(&red[0], &red[0]+3);

		stains.push_back(Blue);		
		stains.push_back(Green);		
		stains.push_back(Red);

		cv::Mat intensity1=Ocv_ColorDeconvolution( stains,img, s);
		vector<cv::Mat> Deconvolved(3);
		split(intensity1, Deconvolved);
		cv::Mat DAB=Deconvolved[1];
		/*
		vector<vector<double>> DABvec;
		for(int j=0; j<DAB.rows; ++j)
		{
			vector<double> DABvec_tmp;
			for(int k=0; k<DAB.cols; ++k)
			{
				DABvec_tmp.push_back(double(Deconvolved[1].at<unsigned char>(j,k)));
			}
			DABvec.push_back(DABvec_tmp);
		}
		*/
		/*
		Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
		cv::Mat img1;
		resize(DAB, img1, img_size);//50% redeuction in display to fit in display-view
		imshow("DAB", img1);
		waitKey(0);
		*/
		//multiScaleFilter2D(I[DAB], type, options) 
		cv::Mat outIm=ocv_multiScaleFilter2D( DAB, s);
		
		//***Save outIm***
		char *extn1=".P.yml"; 
		string file11 = string(imagename2)+string(extn1);
		const char *file22=file11.c_str();
		cv::FileStorage storage1(file22, cv::FileStorage::WRITE);
		storage1 << "P" << outIm;
		storage1.release();
		

	}

	return 0;
}