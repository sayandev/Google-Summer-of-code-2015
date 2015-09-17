#include "stdafx.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <mat.h>

using namespace std;

void matread(const char *file, std::vector<double>& v)
{
    // open MAT-file
    MATFile *pmat = matOpen(file, "r");
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray *arr = matGetVariable(pmat, "bwVesselMask1");
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

int main()
{

	std::vector<double> v;
    matread("output_image.0.seg.mat", v);
	cout<<"Size::"<<v.size()<<endl;
    //for (size_t i=0; i<v.size(); ++i)
        //std::cout << v[i] << std::endl;
	cout<<"Hello World:"<<endl;
	system("pause");

	return 0;
}