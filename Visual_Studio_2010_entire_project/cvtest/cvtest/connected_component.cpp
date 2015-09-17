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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ap.h"

using namespace std;
using namespace cv;
using namespace flann;


void labelBlobs(const cv::Mat &binary, std::vector < std::vector<Point> > &blobs)
{
    blobs.clear();
 
    // Using labels from 2+ for each blob
    Mat label_image;
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



int _tmain(int argc, _TCHAR* argv[])
{
	//char *imagename=&image_name1[0];
	 const char* imagename = "output_image.0.tif";//while reading one image only
	 Mat imgGray;//Initialize a matrix to store the gray version of color image
	 Mat img = imread(imagename);
	 Size s = img.size();
	 cout<<"Image Height:"<<s.height<<endl;
	 cout<<"Image Width:"<<s.width<<endl;
	 Size size(s.height/2,s.width/2);//50% redeuction size initialization
	 cout<<"Value"<<img.ptr<Vec3b>(s.height-1,s.width-1)[0]<<endl;//img.at<Mat>(10,10)//Displaying pixel value
	 Mat img2;
	 resize(img, img2, size);//50% redeuction in display to fit in display-view

	 //Read the corresponding segmentation .mat file
	 std::vector<double> v;
     matread("output_image.0.bmp.mat", v);	 //
     //for (size_t i=0; i<v.size(); ++i)
        //std::cout << v[i] << std::endl; 
	 //convert to mat from vector
	 Mat mymat=Mat(v);
	 cout<<"Size::"<<mymat.size()<<endl;
	 //cout<<mymat<<endl;
	 //resizing the mat
	 Mat mat_dst(s.width,s.height,CV_64FC1);//Read the image size
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
	 //circle( img, Point( 200, 200 ), 1.0, Scalar( 0, 0, 255 ), 1, 8 );
	 //circle( img, Point( 400, 400 ), 1.0, Scalar( 0, 0, 255 ), 1, 8 );
	//CT2CA pszConvertedAnsiString2 (cstr);
	//string image_name_save(pszConvertedAnsiString2);
	//string saveimages =  string(Save_images)+string(image_name_save);
	//imwrite(saveimages, img); 	 
	vector<vector<Point>> blobs;
	labelBlobs(mat_dst,blobs);
	//**Image dialation with 5X5 rectangle structure element
	Mat img341;
	dilate(mat_dst, img341, getStructuringElement(1, Size (5,5)));
	//getStructuringElement(1, Size (5,5));
	
	Mat img34;
	resize(img341, img34, size);
		/*
	
	imshow("image2", img34);
	imwrite("Image_dialation_with_rect_5by5.jpg", img341);
	waitKey(0);
	 	if(img.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imagename);
		return -1;
	}	
		*/
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

	cout << "Bolbsize " << blobs.size() << endl;

	cout <<mat_dst.channels() << endl;

	//drawContours( mat_dst, blobs, -1,CV_RGB(255,255,255),2 );

	vector<vector<Point>> contours;
	Mat img225;
	mat_dst.convertTo(img225,CV_8U);
	findContours(img225,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
	drawContours( mat_dst, contours, -1,CV_RGB(255,255,255),2 );
	//*Convert Binary to color
	Mat img223;
	mat_dst.convertTo(img223,CV_32F);//**Changing binary datatype
	Mat img224;
	cvtColor(img223,img224, CV_GRAY2RGB,3 );//**Changing to 3 channel RGB
	vector<double> area_contour;
	for (int i = 0; i < contours.size(); i++ ) 
	{
		for (int j = 0; j < contours[i].size(); j++ )
		{
			circle( img224, contours[i][j], 1.0, Scalar( 0, 0, 255 ), 1, 8 );//**Plotting the Contours of the Connected Components
		}
		area_contour.push_back(contourArea(contours[i]));
	}
	
	for (int i = 0; i < bolb_center.size(); i++ ) 
	{
		circle( img224, bolb_center[i], 1.0, Scalar( 0, 0, 0 ), 1, 8 );//**Plotting the Centers of the Connected Components
	}
	
	Mat img222;
	resize(img224, img222, size);
	imwrite("vessel_Contour_center_segmented_slice.jpg", img222);
	imshow("image1", img224); 
	waitKey(0);
	 	if(img.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imagename);
		return -1;
	}	
	//vector<vector<Point> > contours;
	//vector<Vec4i> hierarchy;
	//findContours(mat_dst,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE,Point(0, 0));
	//findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Mat img22;
	resize(mat_dst, img22, size);//50% redeuction in display to fit in display-view
	imshow("image1", img22);
	//imwrite("vessel_seg.jpg", mat_dst); 
	 waitKey(0);
	 	if(img.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imagename);
		return -1;
	}	
   
   
	return 0;
}
