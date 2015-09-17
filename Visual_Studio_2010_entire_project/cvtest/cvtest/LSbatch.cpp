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

void matread1(const char *file, std::vector<double>& v)
{
    // open MAT-file
    MATFile *pmat = matOpen(file, "r");
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray *arr = matGetVariable(pmat, "P" );//"name of the variable to be read in the *.mat file"
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

void matread2(const char *file, std::vector<double>& v)
{
    // open MAT-file
    MATFile *pmat = matOpen(file, "r");
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray *arr = matGetVariable(pmat, "ringBW1" );//"name of the variable to be read in the *.mat file"
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

cv::Mat  Heaviside(cv::Mat u, double epsilon)
{
	const double pi = boost::math::constants::pi<double>();
	cv::Mat h,h1;
	h1=u.mul(1/epsilon);
	std::cout<<depthToStr(h1.depth())<< std::endl;
	vector<vector<double>> h1vec;
	for(int j=0; j<h1.rows; ++j)
	{
		vector<double> h1vec_tmp;
		for(int k=0; k<h1.cols; ++k)
		{
			h1vec_tmp.push_back(atan(h1.at<double>(j,k)));
			h1.at<double>(j,k)=atan(h1.at<double>(j,k));
		}
		h1vec.push_back(h1vec_tmp);
	}
	//h1=cv::Mat(h1vec);
	std::cout<<depthToStr(h1.depth())<< std::endl;
	h=0.5*(1+(2/pi)*h1);
	return h;
}

std::vector<cv::Mat > updatef(cv::Mat u,cv::Mat smoothImg,cv::Mat Pp,cv::Mat g, cv::Mat K,double epsilon, double sigma)
{
	vector<cv::Mat > updatedFs;
	Point anchor( 0 ,0 );
	double eps=std::numeric_limits<double>::epsilon();
	cv::Mat  Hu=Heaviside( u, epsilon);
	//cout<<cv::sum( Hu )[0]<<endl;
	cv::Mat N1, N1tmp;
	N1tmp=g.mul(smoothImg);N1tmp=N1tmp.mul(Hu);
	int padsize=((round(2*sigma)*2+1)-1)/2;
	copyMakeBorder(N1tmp,N1tmp,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));	
	filter2D(N1tmp, N1tmp, CV_64FC1 , K,anchor);		
	N1=N1tmp(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
	//cout<<cv::sum( N1 )[0]<<endl;

	cv::Mat D1, D1tmp;
	D1tmp=g.mul(Hu);
	copyMakeBorder(D1tmp,D1tmp,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));	
	filter2D(D1tmp, D1tmp, CV_64FC1 , K,anchor);		
	D1=D1tmp(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
	//cout<<cv::sum( D1 )[0]<<endl;
	D1=D1+eps;
	cv::Mat f1=N1/D1;
	cout<<"f1:"<<cv::sum( f1 )[0]<<endl;
	updatedFs.push_back(f1);		
	cv::Mat N2, N2tmp;
	//cout<<cv::sum( Pp )[0]<<endl;
	cv::Mat Hu1=1-Hu;
	N2tmp=Pp.mul(smoothImg);N2tmp=N2tmp.mul(Hu1);
	copyMakeBorder(N2tmp,N2tmp,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));	
	filter2D(N2tmp, N2tmp, CV_64FC1 , K,anchor);		
	N2=N2tmp(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
	//cout<<cv::sum(N2 )[0]<<endl;

	cv::Mat D2, D2tmp;
	D2tmp=Pp.mul(Hu1);
	copyMakeBorder(D2tmp,D2tmp,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));	
	filter2D(D2tmp, D2tmp, CV_64FC1 , K,anchor);		
	D2=D2tmp(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
	//cout<<cv::sum( D2 )[0]<<endl;

	D2=D2+eps;
	cv::Mat f2=N2/D2;
	cout<<"f2:"<<cv::sum( f2 )[0]<<endl;
	updatedFs.push_back(f2);

	return updatedFs;
	
}

cv::Mat NeumannBoundCond(cv::Mat u)
{
	//NeumannBoundCond
	/*
	vector<vector<double>> myU;
	cv::Mat myU1(5,5,CV_64FC1);
	for(int j=0; j<5; ++j)
	{
		vector<double> myU_tmp;
		for(int k=1; k<6; ++k)
		{
			myU_tmp.push_back(k+5*j);
			myU1.at<double>(j,k-1)=(k+5*j);
		}
		myU.push_back(myU_tmp);
	}
	*/
	cv::Mat fg=u;
	//cout<<fg<<endl;
	int nrow=fg.rows;int ncol=fg.cols;
	//line-1
	fg.at<double>(0,0)=fg.at<double>(2,2);
	fg.at<double>(0,ncol-1)=fg.at<double>(2,ncol-3);
	fg.at<double>(nrow-1,0)=fg.at<double>(nrow-3,2);
	fg.at<double>(nrow-1,ncol-1)=fg.at<double>(nrow-3,ncol-3);
	//cout<<fg<<endl;
	//line-2
	for(int m=1; m<ncol-1; ++m)
	{
		fg.at<double>(0,m)=fg.at<double>(2,m);
		fg.at<double>(nrow-1,m)=fg.at<double>(nrow-3,m);
	}
	//cout<<fg<<endl;
	//line-3
	for(int m=1; m<nrow-1; ++m)
	{
		fg.at<double>(m,0)=fg.at<double>(m,2);
		fg.at<double>(m,ncol-1)=fg.at<double>(m,ncol-3);
	}
	//cout<<fg<<endl;
	return fg;
}
cv::Mat div_norm(cv::Mat u)
{
	cv::Mat ux;
	cv::Sobel(u, ux, CV_64F, 1, 0, 3);
	cv::Mat uy;
	cv::Sobel(u, uy, CV_64F, 0, 1, 3);
	double cnst1=1/(pow(double(10),10));
	cv::Mat normDu=ux.mul(ux)+uy.mul(uy)+cnst1;
	cv::sqrt(normDu,normDu);
	double eps=std::numeric_limits<double>::epsilon();
	normDu=normDu+eps;
	cv::Mat Nx=ux/normDu;
	cv::Mat Ny=uy/normDu;
	cv::Mat px;
	cv::Sobel(Nx, px, CV_64F, 1, 0, 3);
	cv::Mat py;
	cv::Sobel(Ny, py, CV_64F, 0, 1, 3);
	cv::Mat div=px+py;
	return div;

}

cv::Mat  Dirac(cv::Mat u, double epsilon)
{
	const double pi = boost::math::constants::pi<double>();
	double cnst=epsilon/pi;
	cv::Mat u1=pow(epsilon,2)+u.mul(u);
	cv::Mat fd=cnst/u1;
	return fd;
}

cv::Mat  distReg_p2(cv::Mat u)
{
	//distReg_p2
	cv::Mat phi_x;
	cv::Sobel(u, phi_x, CV_64F, 1, 0, 3);
	cv::Mat phi_y;
	cv::Sobel(u, phi_y, CV_64F, 0, 1, 3);
	cv::Mat sr=phi_x.mul(phi_x)+phi_y.mul(phi_y);
	cv::sqrt(sr,sr);
				
	cv::Mat aa(u.size(),CV_64FC1);
	aa.setTo(0);
	cv::Mat mask1=(sr>=0);
	cv::Mat mask2=(sr<=1);
	aa.setTo(1,mask1 & mask2);
	cv::Mat mask3=(sr>1);
	cv::Mat bb(u.size(),CV_64FC1);
	bb.setTo(0);
	bb.setTo(1,mask3);
	cv::Mat ps1,ps2;
	const double pi = boost::math::constants::pi<double>();
	ps1=2*pi*sr;
	for(int k=0; k<ps1.rows; ++k)
	{
		for(int j=0; j<ps1.cols; ++j)
		{
			ps1.at<double>(k,j)=sin(ps1.at<double>(k,j));
		}
	}	 
	ps2=2*pi+bb.mul(sr-1);
	cv::Mat ps=(aa.mul(ps1))/ps2;
	double eps=std::numeric_limits<double>::epsilon();
	cv::Mat dps=ps/(sr+eps);
	cv::Mat dps_x;
	cv::Sobel(dps, dps_x, CV_64F, 1, 0, 3);
	cv::Mat dps_y;
	cv::Sobel(dps, dps_y, CV_64F, 0, 1, 3);
	cv::Mat f1,f2;
	f1=dps_x.mul(phi_x)+dps_y.mul(phi_y);
	Laplacian( u, f2, CV_64F, 3, 1, 0, BORDER_DEFAULT );
	cv::Mat f=f1;//+4*f2;//As laplacian giving error border value

	return f;

}
int _tmain(int argc, _TCHAR* argv[])
{
	cout<<"Let's complete the segmentation ASAP"<<endl;	

	/*
	//Testing Laplacian & del
	vector<vector<double>> myU;
	cv::Mat myU1(5,5,CV_64FC1);
	for(int j=0; j<5; ++j)
	{
		vector<double> myU_tmp;
			for(int k=1; k<6; ++k)
			{
				myU_tmp.push_back(k+5*j);
				myU1.at<double>(j,k-1)=(k+5*j);
			}
		myU.push_back(myU_tmp);
	}
	cv::Mat f2;
	int pdsz=1;
	cv::Mat myU1pad;
	copyMakeBorder(myU1,myU1pad,pdsz,pdsz,pdsz,pdsz,BORDER_REPLICATE);//BORDER_CONSTANT,Scalar(0));	
	
	Laplacian( myU1pad, f2, CV_64F, 3, 1, 0, BORDER_DEFAULT );
	f2=f2(Rect(pdsz,pdsz, myU1.cols,myU1.rows));
	cout<<myU1<<endl;
	cout<<f2<<endl;
	*/

	 CFileFind finder,finder1,finder2;
	 string img_dirPath,img_dirPath1;
	 img_dirPath="C:/Users/Sayan/Dropbox/GSoC_2015/Load_in_second_phase_seg/P_mat/";
	 img_dirPath1="C:/Users/Sayan/Dropbox/GSoC_2015/Load_in_second_phase_seg/ring_bw/";

	 CFileFind finder3;
	 string img_dirPath3;
	 img_dirPath3="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/data_to_segment/";

	 CString  dirPathC31,dirPathC32;
	 CString cstr3,cstrf3;	 	 	 
	 dirPathC31 = img_dirPath3.c_str();
	 dirPathC32 = dirPathC31 + "\\*.bmp";
	 BOOL bWorking2 = finder2.FindFile(dirPathC32);


	 //Reading number of files in the directory
	 int num_img=0;
	 DIR *dir;
	 struct dirent *ent;
	 
	 if ((dir = opendir ("C:\\Users\\Sayan\\Dropbox\\GSoC_2015\\Load_in_second_phase_seg\\P_mat\\")) != NULL) {
		/* print all the files and directories within directory */
		 while ((ent = readdir (dir)) != NULL) {
		 //printf ("%s\n", ent->d_name);
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
	 dirPathC = dirPathC1 + "\\*.mat";
	 BOOL bWorking = finder.FindFile(dirPathC);

	 CString dirPath_21,dirPathC22,dirPathC21;
	 CString cstr2,cstrf2;	 	 	 
	 dirPathC21 = img_dirPath1.c_str();
	 dirPathC22 = dirPathC21 + "\\*.mat";
	 BOOL bWorking1 = finder1.FindFile(dirPathC22);


	 while (bWorking & bWorking1 & bWorking2)

	{
		//Read the corresponding .mat file (P_mat)
		CString dirPathC2;
		dirPathC2="C:/Users/Sayan/Dropbox/GSoC_2015/Load_in_second_phase_seg/P_mat/";
		bWorking = finder.FindNextFile();
		cstr = finder.GetFileName();
		CStringA cstr1(cstr);
		CString finpath=dirPathC1+cstr;
		_tprintf_s(_T("%s\n"), (LPCTSTR) finpath);
		_tprintf_s(_T("%s\n"), (LPCTSTR) cstr);

		//Read the corresponding .mat file (ringbw.mat)
		CString dirPathC23;
		dirPathC23="C:/Users/Sayan/Dropbox/GSoC_2015/Load_in_second_phase_seg/ring_bw/";
		bWorking1 = finder1.FindNextFile();
		cstr2 = finder1.GetFileName();
		CStringA cstr27(cstr2);
		CString finpath1=dirPathC21+cstr2;
		_tprintf_s(_T("%s\n"), (LPCTSTR) finpath1);
		_tprintf_s(_T("%s\n"), (LPCTSTR) cstr2);

		//Read the corresponding Color Image
		CString dirPathC33;
		dirPathC33="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/data_to_segment/";
		bWorking2 = finder2.FindNextFile();
		cstr3 = finder2.GetFileName();
		CStringA cstr37(cstr3);
		CString finpath3=dirPathC31+cstr3;
		_tprintf_s(_T("%s\n"), (LPCTSTR) finpath3);
		_tprintf_s(_T("%s\n"), (LPCTSTR) cstr3);

		dirPathC33=dirPathC33+cstr3;
	    CT2CA pszConvertedAnsiString (dirPathC33);
	    string image_name1(pszConvertedAnsiString);
		char *imagename=&image_name1[0];
		cv::Mat imgGray;//Initialize a matrix to store the gray version of color image
		cv::Mat img = imread(imagename); //Load the registered Image
		cout<<depthToStr(img.depth())<<endl;
		Size s = img.size();
		cvtColor(img, imgGray, CV_RGB2GRAY);
		//Display the 50% Reduction of the loaded image
		cout<<"Image Height:"<<s.height<<endl;
		cout<<"Image Width:"<<s.width<<endl;
		Size img_size(s.height/2,s.width/2);//50% redeuction size initialization
		/*
		cv::Mat img1;
		resize(imgGray, img1, img_size);//50% redeuction in display to fit in display-view
		imshow("Gray_Image", img1);	
		waitKey(5);
		*/

		cstrf=cstr;
		dirPathC2=dirPathC2+cstrf;
		CT2CA pszConvertedAnsiString11 (dirPathC2);	 
		string seg_file(pszConvertedAnsiString11);
		char *seg_file1=&seg_file[0];

		std::vector<double> v;
		matread1(seg_file1, v);
		cv::Mat mymat=cv::Mat(v);
		cout<<"Size::"<<mymat.size()<<endl;
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
		mat_dst=mat_dst.t();
		cv::Mat P=mat_dst;
		cout<<"Transpose::"<<mat_dst.size()<<endl;
		/*
		cv::Mat img2;
		resize(P, img2, img_size);//50% redeuction in display to fit in display-view
		imshow("P_mat", img2);	
		waitKey(5);
		*/

		cstrf2=cstr2;
		dirPathC23=dirPathC23+cstrf2;
		CT2CA pszConvertedAnsiString22 (dirPathC23);	 
		string seg_file2(pszConvertedAnsiString22);
		char *seg_file21=&seg_file2[0];
		std::vector<double> v1;
		matread2(seg_file21, v1);
		cv::Mat mymat1=cv::Mat(v1);
		cout<<"Size::"<<mymat1.size()<<endl;
		cv::Mat mat_dst1(s.width,s.height,CV_64FC1);//Read the image size
		k=0;
		for(int i=0; i<s.width; ++i)
		{
			for(int j=0; j<s.height; ++j)
			{
				mat_dst1.at<double>(i,j)=mymat1.at<double>(k);
				k=k+1;
			}
		}	 
		cout<<"ReSize::"<<mat_dst1.size()<<endl;
		//Transpose the mat
		mat_dst1=mat_dst1.t();
		cv::Mat ringBW=mat_dst1;
		cout<<"Transpose::"<<mat_dst1.size()<<endl;
		/*
		cv::Mat img3;
		resize(ringBW, img3, img_size);//50% redeuction in display to fit in display-view
		imshow("ringbw", img3);	
		waitKey(0);
		*/
		//Done reading all the 3 files
		imgGray.convertTo(imgGray,CV_64FC1);
		double min, max;
		minMaxLoc(imgGray, &min, &max);
		cout<<min<<max<<endl;
		double A=255;

		normalize(imgGray, imgGray, min, max,NORM_MINMAX);

		double alpha=0.001*pow(A,2);// coefficient of arc length term
		double timestep=0.1;
		double beta=0.2/timestep;  // coefficient for distance regularization term (regularize the level set function)
		int gamma = 5;
		int sigma = 4; // scale parameter that specifies the size of the neighborhood
		int lambda1 = 1;
		int lambda2 = 1;
		int iter_outer=50;//50; //***** to be updated
		int iter_inner=10;   // inner iteration for level set evolution
		double epsilon=1;   //for Heaviside function
		int c0=5;

		cv::Mat initialLSF(s,CV_64FC1);
		initialLSF.setTo(1);
		initialLSF=-1*c0*initialLSF;
		cv::Mat mask=(ringBW==1);
		initialLSF.setTo(c0,mask);
		cv::Mat u=initialLSF;
		//cout<<cv::sum( initialLSF )[0]<<endl;
		//Matlab: K=fspecial('gaussian',round(2*sigma)*2+1,sigma); % Next 3 lines
		cv::Mat KX= getGaussianKernel( round(2*sigma)*2+1, sigma,  CV_64F);
		cv::Mat KY = getGaussianKernel(round(2*sigma)*2+1, sigma,  CV_64F); 
		cv::Mat K = KX * KY.t(); 

		Point anchor( 0 ,0 );
		cv::Mat KONE(s,CV_64FC1);
		KONE.setTo(1);
		cv::Mat KONEpadded;
		//Number of padding {[size(lingrid)-1]/2}
		int padsize=((round(2*sigma)*2+1)-1)/2;
		copyMakeBorder(KONE,KONEpadded,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));

		filter2D(KONEpadded, KONEpadded, CV_64FC1 , K,anchor);
		
		KONE=KONEpadded(Rect(0, 0, KONE.cols,KONE.rows));
		//cout<<cv::sum( KONE )[0]<<endl;
		

		vector <double> krtmp;
		double myints[] = {1,0,-1};
		krtmp.assign(myints,myints+3);
		cv::Mat krnl=cv::Mat(krtmp);
		
		cv::Mat krnltrns=krnl.t();
		cout<<krnl<<endl;
		padsize=krnl.rows;
		padsize=(padsize-1)/2;
		cv::Mat Kx;
		cv::Mat Kpadded;
		copyMakeBorder(K,Kpadded,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));
		filter2D(Kpadded, Kx, CV_64FC1 , krnltrns,anchor);
		Kx=Kx(Rect(padsize, padsize, K.cols-(padsize+1),K.rows));
		cv::Mat Ky;
		filter2D(Kpadded, Ky, CV_64FC1 , krnl,anchor);
		Ky=Ky(Rect(padsize, padsize, K.cols ,K.rows-(padsize+1)));
		
		Kx=-1*Kx;
		Ky=-1*Ky;

		

		/*
		vector<vector<double>> Kxvec;
		for(int j=0; j<Kx.rows; ++j)
		{
			vector<double> Kxvec_tmp;
			for(int k=0; k<Kx.cols; ++k)
			{
				Kxvec_tmp.push_back(Kx.at<double>(j,k));
			}
			Kxvec.push_back(Kxvec_tmp);
		}

		vector<vector<double>> Kyvec;
		for(int j=0; j<Ky.rows; ++j)
		{
			vector<double> Kyvec_tmp;
			for(int k=0; k<Ky.cols; ++k)
			{
				Kyvec_tmp.push_back(Ky.at<double>(j,k));
			}
			Kyvec.push_back(Kyvec_tmp);
		}
		*/
		cv::Mat fx;
		cv::Mat fy;
		filter2D(imgGray, fx, CV_64FC1 , Kx,anchor);
		filter2D(imgGray, fy, CV_64FC1 , Ky,anchor);
		/*
		cout<<cv::sum( fx )[0]<<endl;
		cout<<cv::sum( fy )[0]<<endl;

		cv::Mat img3;
		resize(fx, img3, img_size);//50% redeuction in display to fit in display-view
		imshow("fx", img3);	
		waitKey(5);

		cv::Mat img4;
		resize(fy, img4, img_size);//50% redeuction in display to fit in display-view
		imshow("fy", img4);	
		waitKey(0);
		*/
		double r=1.5;
		cv::Mat g=fx.mul(fx)+fy.mul(fy);
		g=(-0.5*g)/pow(r,2);
		cv::exp(g,g);
		/*
		cv::Mat img5;
		resize(g, img5, img_size);//50% redeuction in display to fit in display-view
		imshow("g", img5);	
		waitKey(0);
		*/
		int sigmaIm = 1; 
		cv::Mat KXx= getGaussianKernel( round(2*sigmaIm)*2+1, sigmaIm,  CV_64F);
		cv::Mat KYy = getGaussianKernel(round(2*sigmaIm)*2+1, sigmaIm,  CV_64F); 
		cv::Mat Kk = KXx * KYy.t(); 

		cv::Mat smoothImg;
		cv::Mat Imgpadded;
		//Number of padding {[size(lingrid)-1]/2}
		padsize=((round(2*sigmaIm)*2+1)-1)/2;
		copyMakeBorder(imgGray,Imgpadded,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));
		filter2D(Imgpadded, smoothImg, CV_64FC1 , Kk,anchor);		
		smoothImg=smoothImg(Rect(0, 0, imgGray.cols,imgGray.rows));
		//cout<<cv::sum( smoothImg )[0]<<endl;
		cv::Mat Pp(s,CV_64FC1);
		Pp.setTo(1);//My P for function lse
		for(int i=0; i<iter_outer; ++i)
		{
			//[u, f1, f2]= lse(u, smoothImg, P, g, K,      KONE, lambda1, lambda2, alpha, beta, gamma, timestep, epsilon, iter_inner);
			                //(u0, Img,      P, g, Ksigma, KONE, lambda1, lambda2, alpha, beta, gamma, timestep, epsilon, iter_lse)
			//Only u used after funtion return

		//lse start
			vector<cv::Mat > updtd_Fs= updatef( u, smoothImg, Pp, g,  K, epsilon, sigma);

			cv::Mat f1=updtd_Fs[0];
			cv::Mat f2=updtd_Fs[1];

			//updateLSF start
			//updateLSF(u, Img, Ksigma, KONE, f1, f2, P, g, lambda1, lambda2, alpha, beta, gamma, timestep, epsilon, iter_lse);
			cv::Mat e1,e1t1,e1t2,e1t2f,e1t3,e1t3f;
			e1t1=smoothImg.mul(smoothImg);e1t1=e1t1.mul(KONE);			
			padsize=((round(2*sigma)*2+1)-1)/2;
			e1t2=f1;
			copyMakeBorder(e1t2,e1t2,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));	
			filter2D(e1t2, e1t2, CV_64FC1 , K,anchor);		
			e1t2f=e1t2(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
			e1t3=f1.mul(f1);
			copyMakeBorder(e1t3,e1t3,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));	
			filter2D(e1t3, e1t3, CV_64FC1 , K,anchor);	
			e1t3f=e1t3(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
			cv::Mat smoothImg1=2*smoothImg;
			e1=e1t1-smoothImg1.mul(e1t2f)+e1t3f;
			cout<<"e1:"<<cv::sum( e1 )[0]<<endl;
			cv::Mat e2,e2t2,e2t2f,e2t3,e2t3f;
			e2t2=f2;
			copyMakeBorder(e2t2,e2t2,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));	
			filter2D(e2t2, e2t2, CV_64FC1 , K,anchor);		
			e2t2f=e2t2(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
			e2t3=f2.mul(f2);
			copyMakeBorder(e2t3,e2t3,padsize,padsize,padsize,padsize,BORDER_CONSTANT,Scalar(0));
			filter2D(e2t3, e2t3, CV_64FC1 , K,anchor);
			e2t3f=e2t3(Rect(padsize, padsize, smoothImg.cols,smoothImg.rows));
			e2=e1t1-smoothImg1.mul(e2t2f)+e2t3f;
			cout<<"e2:"<<cv::sum( e2 )[0]<<endl;
			for(int l=0; l<iter_inner; ++l)
			{	
				u=NeumannBoundCond(u);
				cv::Mat kdiv= div_norm(u);
				cv::Mat DiracU=Dirac( u,epsilon);
				cv::Mat ImageTerm=-1*DiracU.mul((lambda1*e1).mul(g)-(lambda2*e2).mul(Pp));

				cv::Mat gx;
				cv::Sobel(g, gx, CV_64F, 1, 0, 3);
				cv::Mat gy;
				cv::Sobel(g, gy, CV_64F, 0, 1, 3);
				cv::Mat ux;
				cv::Sobel(u, ux, CV_64F, 1, 0, 3);
				cv::Mat uy;
				cv::Sobel(u, uy, CV_64F, 0, 1, 3);
				double cnst1=1/(pow(double(10),10));
				cv::Mat normDu=ux.mul(ux)+uy.mul(uy)+cnst1;
				cv::sqrt(normDu,normDu);
				double eps=std::numeric_limits<double>::epsilon();
				normDu=normDu+eps;
				cv::Mat Nx=ux/normDu;
				cv::Mat Ny=uy/normDu;

				cv::Mat lengthTerm= (alpha*DiracU).mul(gx.mul(Nx)+gy.mul(Ny)+g.mul(kdiv));
				cv::Mat ff=  distReg_p2(u);
				cv::Mat penalizeTerm=beta*ff;
				cv::Mat areaTerm =-gamma*DiracU.mul(1-P);
				cout<<"areaTerm:"<<cv::sum( areaTerm )[0]<<endl;

				u=u+timestep*(ImageTerm+lengthTerm+penalizeTerm+areaTerm);	
				//updateLSF end
			//lse end
			}

		}

		cv::Mat img5;
		resize(u, img5, img_size);//50% redeuction in display to fit in display-view
		imshow("u", img5);	
		waitKey(5);

		CT2CA pszConvertedAnsiString1 (cstr);
	    string image_name11(pszConvertedAnsiString1);
		char *imagename2=&image_name11[0];
		char *extn=".seg.yml"; 
		string file1 = string(imagename2)+string(extn);
		const char *file2=file1.c_str();
		cv::FileStorage storage(file2, cv::FileStorage::WRITE);
		storage << "u" << u;
		storage.release();




	}

	return 0;
}