#ifndef _BASE_FUNC_H
#define _BASE_FUNC_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int myorderfilter(cv::Mat& curvedness, cv::Mat& Output);

//Mat retmat(cv::Mat& Inmat, cv::Mat& Outmat)

#endif