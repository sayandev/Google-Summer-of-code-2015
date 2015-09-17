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
#include <ctype.h>

//before VTK includes
#define vtkRenderingCore_AUTOINIT 4(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingFreeTypeOpenGL,vtkRenderingOpenGL)
#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)


#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkMarchingCubes.h>
#include <vtkVoxelModeller.h>
#include <vtkSphereSource.h>
#include <vtkImageData.h>
#include <vtkDICOMImageReader.h>
#include "vtkVolume16Reader.h"
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

#define VTK_DATA_ROOT "S:\\\VTK\\\Data\\VTKData\\\Data\\"  


using namespace std;
using namespace cv;
using namespace flann;
//using namespace METAIO_NAMESPACE;
int _tmain(int argc, _TCHAR* argv[])

{
	cout<<"***Lets Run VTK***"<<endl;

	vtkSmartPointer<vtkImageData> volume =
    vtkSmartPointer<vtkImageData>::New();
	double isoValue;

	//vtkSmartPointer<vtkDICOMImageReader> reader =
      //vtkSmartPointer<vtkDICOMImageReader>::New();
   // reader->SetDirectoryName("S:\\GSoC_2015_Implementation\\temp1\\");

	//Read the image
	vtkVolume16Reader *reader = vtkVolume16Reader::New();
	reader->SetDataDimensions(64,64);
	reader->SetDataByteOrderToBigEndian();
	reader->SetFilePrefix("S:\\GSoC_2015_Implementation\\temp1\\Image");
	reader->SetImageRange(1, 551);
	reader->SetDataSpacing( 3.2, 3.2, 1.5);
    reader->Update();
    volume->DeepCopy(reader->GetOutput());
    isoValue = 0.1;

	vtkSmartPointer<vtkMarchingCubes> surface = 
    vtkSmartPointer<vtkMarchingCubes>::New();
 
	#if VTK_MAJOR_VERSION <= 5
	  surface->SetInput(volume);
	#else
	  surface->SetInputData(volume);
	#endif
	  surface->ComputeNormalsOn();
	  surface->SetValue(0, isoValue);
 
	  vtkSmartPointer<vtkRenderer> renderer = 
		vtkSmartPointer<vtkRenderer>::New();
	  renderer->SetBackground(.1, .2, .3);
 
	  vtkSmartPointer<vtkRenderWindow> renderWindow = 
		vtkSmartPointer<vtkRenderWindow>::New();
	  renderWindow->AddRenderer(renderer);
	  vtkSmartPointer<vtkRenderWindowInteractor> interactor = 
		vtkSmartPointer<vtkRenderWindowInteractor>::New();
	  interactor->SetRenderWindow(renderWindow);
 
	  vtkSmartPointer<vtkPolyDataMapper> mapper = 
		vtkSmartPointer<vtkPolyDataMapper>::New();
	  mapper->SetInputConnection(surface->GetOutputPort());
	  mapper->ScalarVisibilityOff();
 
	  vtkSmartPointer<vtkActor> actor = 
		vtkSmartPointer<vtkActor>::New();
	  actor->SetMapper(mapper);
 
	  renderer->AddActor(actor);
 
	  renderWindow->Render();
	  interactor->Start();

	return 0;
}