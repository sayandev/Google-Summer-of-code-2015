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

int _tmain(int argc, _TCHAR* argv[])
{	
	cout<<"Let's confirm the OCV functions"<<endl;

	cv::Mat M = cv::Mat(3, 3, CV_32SC1);
	randu(M,Scalar(10),Scalar(1));
	std::cout<<depthToStr(M.depth())<< std::endl;
	std::cout<<M<< std::endl;

	cv::Mat M1 = cv::Mat(3, 3, CV_64FC1);
	randu(M1,Scalar(10),Scalar(1));
	std::cout<<M1<< std::endl;

	cv::Mat M3 = cv::Mat(3, 3, CV_64FC1);
	randu(M3,Scalar(10),Scalar(1));
	std::cout<<M3<< std::endl;

	cv::Mat M2 = cv::Mat(3, 3, CV_32SC1);
	randu(M2,Scalar(10),Scalar(1));
	std::cout<<M2<< std::endl;

	//cout<<abs(M1-M3)<<endl;

	//cv::Mat check = (cv::abs(M1-M3)>0);//good for Int
	cv::Mat check = (cv::abs(M1)>abs(M3));
	cout<<check<<endl;
	//M1.setTo(1, check);//works
	M3.copyTo(M1,check);//works
	std::cout<<M1<< std::endl;


	//std::cout<<M1.mul(1/M)<< std::endl; //.mul & .mul(1/) tested

	//std::cout<<M.mul(M)<< std::endl;
	//std::cout<<M1.mul(M1)<< std::endl; //Square tested

	//std::cout<<-1*M<< std::endl;
	//std::cout<<M1.mul(-1)<< std::endl;//Multiply with '-1' both work
	//std::cout<<M1-0.5*(M1+M1)<< std::endl;//addition, substarct, multiplication with scalar
	cv::Mat tmp,tmp1;
	//cv::sqrt(M1,tmp1);
	//cout<<tmp1<<endl;//sqrt tested only work on CV_64F 
	//cv::exp(M,tmp1);
	//cout<<tmp1<<endl;//exp tested only work on CV_64F 

	//cv::add(M1,M1,tmp,noArray(),CV_64FC1);
	//cout<<tmp<<endl;
	//cout<<M1+M1<<endl;
	//cv::subtract(M1+M1,tmp,tmp1,noArray(),CV_64FC1);
	//cout<<tmp1<<endl;//Add & Substract work both way for double & float

	//cout<<M1/M1<<endl;
	//cout<<M1.mul(1/M1)<<endl;//element by element division both works
	

	return 0;
}