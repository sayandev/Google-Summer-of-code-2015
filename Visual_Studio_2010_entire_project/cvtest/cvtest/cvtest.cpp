// cvtest.cpp : Defines the entry point for the console application.
//

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
#include <valarray> 
#include <complex>
#define BUFSIZE 256
const double PI = 3.141592653589793238460;
 
typedef unsigned long int uint32_t;

typedef std::complex<double> Complex_fft;
typedef std::valarray<Complex_fft> CArray_fft;

//#include "cvblob.h"

using namespace std;
using namespace cv;
using namespace flann;
//using namespace cvb;
using namespace alglib;
using namespace arma;
using namespace alglib_impl;

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

std::vector<double> linspace(double first, double last, int len) {
  std::vector<double> result(len);
  double step = (last-first) / (len - 1);
  for (int i=0; i<len; i++) { result[i] = first + i*step; }
  return result;
}



void labelBlobs(const cv::Mat &binary, std::vector < std::vector<Point> > &blobs)
{
    blobs.clear();
 
    // Using labels from 2+ for each blob
    cv::Mat label_image;
    binary.convertTo(label_image, CV_32FC1);
 
    int label_count = 2; // starts at 2 because 0,1 are used already
 
    for(int y=0; y < binary.rows; y++) {
        for(int x=0; x < binary.cols; x++) {
            if((int)label_image.at<float>(y,x) != 1) {
                continue;
            }
 
            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);
 
            std::vector<Point> blob;
 
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if((int)label_image.at<float>(i,j) != label_count) {
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

int matwrite(cv::Mat myU1, const char *file, const char *varname)
{
	myU1=myU1.t();
	double const *testData1D = (double*)myU1.data;
	vector<double> values(testData1D, testData1D + 9);
	double rawdata[9];
	for(int j=0; j<9; ++j)
	{
		rawdata[j]=values[j];
	}
		
	//
	MATFile *pmat;
	mxArray *pa1, *pa2, *pa3;
	std::vector<int> myInts;
	myInts.push_back(1);
	myInts.push_back(2);
	printf("Accessing a STL vector: %d\n", myInts[1]);

	
	char str[BUFSIZE];
	int status; 

	printf("Creating file %s...\n\n", file);
	pmat = matOpen(file, "w");
	if (pmat == NULL) {
	printf("Error creating file %s\n", file);
	printf("(Do you have write permission in this directory?)\n");
	return(EXIT_FAILURE);
	}

	pa1 = mxCreateDoubleMatrix(3,3,mxREAL);
	if (pa1 == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	pa2 = mxCreateDoubleMatrix(3,3,mxREAL);
	if (pa2 == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}
	//memcpy((void *)(mxGetPr(pa2)), (void *)data, sizeof(data));
	memcpy((void *)(mxGetPr(pa2)), (void *)rawdata, sizeof(rawdata));
  
	pa3 = mxCreateString("MATLAB: the language of technical computing");
	if (pa3 == NULL) {
		printf("%s :  Out of memory on line %d\n", __FILE__, __LINE__);
		printf("Unable to create string mxArray.\n");
		return(EXIT_FAILURE);
	}

	status = matPutVariable(pmat, "LocalDouble", pa1);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}  
  
	status = matPutVariableAsGlobal(pmat, varname, pa2);
	if (status != 0) {
		printf("Error using matPutVariableAsGlobal\n");
		return(EXIT_FAILURE);
	} 
  
	status = matPutVariable(pmat, "LocalString", pa3);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	} 

	/*
	   * Ooops! we need to copy data before writing the array.  (Well,
	   * ok, this was really intentional.) This demonstrates that
	   * matPutVariable will overwrite an existing array in a MAT-file.
	   */
	  memcpy((void *)(mxGetPr(pa1)), (void *)rawdata, sizeof(rawdata));
	  //memcpy((void *)(mxGetPr(pa1)), (void *)data, sizeof(data));
	  status = matPutVariable(pmat, "LocalDouble", pa1);
	  if (status != 0) {
		  printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		  return(EXIT_FAILURE);
	  } 
  
	  /* clean up */
	  mxDestroyArray(pa1);
	  mxDestroyArray(pa2);
	  mxDestroyArray(pa3);

	if (matClose(pmat) != 0) 
	{
		printf("Error closing file %s\n",file);
		return(EXIT_FAILURE);
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	//test to write mat files
	
	  cout<<"Lets write everything as.mat"<<endl;
	  //
	    vector<vector<double>> myU;
		cv::Mat myU1(3,3,CV_64FC1);
		for(int j=0; j<3; ++j)
		{
			vector<double> myU_tmp;
			for(int k=1; k<4; ++k)
			{
				myU_tmp.push_back(k+5*j);
				myU1.at<double>(j,k-1)=(k+3*j);
			}
			myU.push_back(myU_tmp);
		}
		cout<<myU1<<endl;
	const char *file = "mattest.mat";	
	const char *varname="GlobalDouble";
	matwrite(myU1, file,varname);
	  



	//Upto here
	 CFileFind finder;
	 string img_dirPath;
	 img_dirPath="C:/Users/Sayan/Dropbox/GSoC_2015_Implementation/Registered_images/";
	 CString dirPath_1,dirPathC,dirPathC1;
	 CString cstr,cstrf;	 	 	 
	 dirPathC1 = img_dirPath.c_str();
	 dirPathC = dirPathC1 + "\\*.tif";
	 BOOL bWorking = finder.FindFile(dirPathC);
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
	 cstrf=cstr;
	 cstrf.Delete(cstrf.GetLength()-3,3);
	 dirPath_1=dirPath_1+cstrf;
	 CT2CA pszConvertedAnsiString1 (dirPath_1);	 
	 string seg_file(pszConvertedAnsiString1);
	 seg_file.append("bmp.mat");
	 char *seg_file1=&seg_file[0];
	 //int curPos = 0;
	 //CString resToken= cstr.Tokenize(_T("."),curPos);
	 //CStringA finpath1(finpath);
	 //CStringA image_name=cstr1.Left(8);
	 dirPathC2=dirPathC2+cstr;
	 CT2CA pszConvertedAnsiString (dirPathC2);
	 string image_name1(pszConvertedAnsiString);
	
	 char *imagename=&image_name1[0];
	 //const char* imagename = "output_image.0.tif";//while reading one image only
	 cv::Mat imgGray;//Initialize a matrix to store the gray version of color image
	 cv::Mat img = imread(imagename);
	 Size s = img.size();
	 cout<<"Image Height:"<<s.height<<endl;
	 cout<<"Image Width:"<<s.width<<endl;
	 Size size(s.height/2,s.width/2);//50% redeuction size initialization
	 cout<<"Value"<<img.ptr<Vec3b>(s.height-1,s.width-1)[0]<<endl;//img.at<Mat>(10,10)//Displaying pixel value
	 
	 
	
	 //Read the corresponding segmentation .mat file
	 std::vector<double> v;
     matread(seg_file1, v);	 //"output_image.0.bmp.mat"
     //for (size_t i=0; i<v.size(); ++i)
        //std::cout << v[i] << std::endl; 
	 //convert to mat from vector
	 cv::Mat mymat=cv::Mat(v);
	 cout<<"Size::"<<mymat.size()<<endl;
	 //cout<<mymat<<endl;
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
	 //cout<<mat_dst<<endl;
	 //Transpose the mat
	 mat_dst=mat_dst.t();
	 cout<<"Transpose::"<<mat_dst.size()<<endl;
	 //cout<<mat_dst<<endl;
	 cout<<sum(mat_dst)[0]<<endl;
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
	cout<<sizeof(row)<<endl;
	cout<<sizeof(col)<<endl;
	cout<<row[0]<<endl;
	cout<<row[count-1]<<endl;
	cout<<col[0]<<endl;
	cout<<col[count-1]<<endl;

	 //Display
	for (int i = 0; i < count; i++ ) 
	{
		circle( img, Point( col[i], row[i] ), 1.0, Scalar( 0, 0, 255 ), 1, 8 );
	}
	cv::Mat img341;
	dilate(mat_dst, img341, getStructuringElement(MORPH_RECT  , Size (5,5)));//Perform Image dialation on the RGB image
	/* 
	Mat img2;
	resize(img341, img2, size);//50% redeuction in display to fit in display-view
	 imshow("image1", img2);	 
	 waitKey(0);
	 	if(img.empty())
	    {
		   fprintf(stderr, "Can not load image %s\n", imagename);
		   return -1;
	    }	
	*/

	vector<vector<Point>> blobs;
	cv::Mat img225;
	img341.convertTo(img225,CV_8U);
	findContours(img225,blobs,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);//*** Find contour matches with Matlab better then labelBolbs
	vector<vector<Point>> blobs1;
	labelBlobs(img225,blobs1);
	//**Find the center of the bolbs
	vector<Point> bolb_center;
	vector<Point> bolb_center1;
	vector<double> bolb_area;
	
	for (int i = 0; i < blobs.size(); i++ ) {
		float sum_x=0;
		float sum_y=0;
		for (int j = 0; j < blobs[i].size(); j++) {
			sum_x=sum_x+blobs[i][j].x;
			sum_y=sum_y+blobs[i][j].y;			
		}
		bolb_center.push_back(Point((int)sum_x/blobs[i].size(),(int)sum_y/blobs[i].size()));//**bolb center by average up the pixel location
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
	//sort(bolb_area.begin(), bolb_area.end(), greater<int>());
	vector<Point> bolb_center_sorted;
	vector<Point> bolb_center1_sorted;
	for (int i = 0; i < bolb_area.size(); i++ ) {//** Guess Not top 30 but choose the points whose area greater then 30
		bolb_center_sorted.push_back(bolb_center.at(vp[i].second));
		bolb_center1_sorted.push_back(bolb_center1.at(vp[i].second));//*** Better match with Matlab version
		if (vp[i].first<=80)
			break;
	}
	vector<vector<Point>> blobs_sorted;
	for (int i = 0; i < bolb_area.size(); i++ ) {//** Guess Not top 30 but choose the points whose area greater then 30
		blobs_sorted.push_back(blobs.at(vp[i].second));//*** Better match with Matlab version
		if (vp[i].first<=80)
			break;
	}
	cv::Mat img226;
	cvtColor(img225,img226, CV_GRAY2RGB,3 );
	//cout<<blobs_sorted[0].size()<<endl;
	cout<<blobs_sorted[0][0].x<<endl;
	//**Plot test for max 30 area boundaries
	/*
	for (int i = 0; i < blobs_sorted[29].size(); i++ ) 
	{
		circle( img226,  blobs_sorted[29][i], 1.0, Scalar( 0, 0, 255 ), 1, 8 );
	}
	
	Mat img227;
	resize(img226, img227, size);
	imshow("image1", img227);	 
	 waitKey(0);
	 	if(img.empty())
	    {
		   fprintf(stderr, "Can not load image %s\n", imagename);
		   return -1;
	    }	

	*/

	//**Test CV bolb (unsuccessful)
	/*
	//CvContour();
	CvBlobs blobs;
	IplImage* image262=cvCloneImage(&(IplImage)img225);
	IplImage* image263;
	unsigned int result = cvLabel(image262, image263, blobs);

   //vector< pair<CvLabel, CvBlob*> > blobList;
   //copy(blobs.begin(), blobs.end(), back_inserter(blobList));

  //sort(blobList.begin(), blobList.end(), cmpY);

  //for (int i=0; i<blobList.size(); i++)
  //{
    // This will print: [LABEL] -> BLOB_DATA
   // cout << "[" << blobList[i].first << "] -> " << (*blobList[i].second) << endl;
 // }
	*/
	 //circle( img, Point( 200, 200 ), 1.0, Scalar( 0, 0, 255 ), 1, 8 );
	 //circle( img, Point( 400, 400 ), 1.0, Scalar( 0, 0, 255 ), 1, 8 );

	//***Interpolation for getting 64 points around the bolb boundary
	int num_intrp=65;
	std::vector<double> grid1 = linspace(0, 1.0, num_intrp);//Equivalent to (0:1/n-1:1) in matlab
	std::vector<double> X, Y;
	 for(int i=0; i<blobs_sorted[0].size(); i++){
		 X.push_back(blobs_sorted[0][i].x);
		 Y.push_back(blobs_sorted[0][i].y);
    }
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
	/*
	for ( int j = 0; j < tbins.size()-1; ++j)
	{
		for (int k=0;k<(int)double(z.at(j)); k++)
		{
			tbins.at(count11)=j+1;
			count11=count11+1;
		}
	}
	*/
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
	//Use the area of the measured by the contour detected in find contour no need to compute area using the interpolated points

	/*
	cv::Mat img227;
	resize(img226, img227, size);
	imshow("image1", img227);	 
	 waitKey(0);
	 	if(img.empty())
	    {
		   fprintf(stderr, "Can not load image %s\n", imagename);
		   return -1;
	    }
	*/
	//***Alglib spline (Not working need to figure how to solve aserror type by including "alglib_impl::ae_state *_state;")
	/*
	vector<double> intrpltd_y(sum_of_z);
    alglib::real_1d_array AX, AY;
    AX.setcontent(X.size(), &(X[0]));
    AY.setcontent(Y.size(), &(Y[0]));
	alglib::spline1dinterpolant spline;
	alglib_impl::ae_state *_state;//***
	alglib::spline1dbuildlinear(AX, AY, spline);
	for(int i=0; i<num_intrp; i++){
      double x11=eq_spaced_x[i];
      intrpltd_y.at(i)=alglib::spline1dcalc(spline,x11);
    }
	*/
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
	CT2CA pszConvertedAnsiString2 (cstr);
	string image_name_save(pszConvertedAnsiString2);
	string saveimages =  string(Save_images)+string(image_name_save);
	imwrite(saveimages, img); 	 
	/*
	imshow("image1", img);	 
	 waitKey(0);
	 	if(img.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imagename);
		return -1;
	}	
	*/
	img.release();
  }
   
    
	return 0;
}





