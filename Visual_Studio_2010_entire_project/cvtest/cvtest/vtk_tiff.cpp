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

//Include files for the prelim VTK emxample
/*
#include "vtkCylinderSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkProperty.h"
#include "vtkCamera.h"
*/

/*
#include <vtkMetaImageReader.h>
#include <vtkImageAccumulate.h>
#include <vtkDiscreteMarchingCubes.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkMaskFields.h>
#include <vtkThreshold.h>
#include <vtkGeometryFilter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSmartPointer.h>
 
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtksys/ios/sstream>
#include <vtkmetaio/metaObject.h>
*/

#include "vtkActor.h"
#include "vtkCamera.h"
#include "vtkContourFilter.h"
#include "vtkOutlineFilter.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataNormals.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkVolume16Reader.h"
#include "vtkVolumeReader.h"
#include "vtkImageReader2.h"
#include "vtkSmartPointer.h"
#include <vtkImageViewer2.h>
#include <vtkTIFFReader.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>

#include "vtkProperty.h"
#include "vtkCamera.h"

#define VTK_DATA_ROOT "S:\\\VTK\\\Data\\VTKData\\\Data\\"  


using namespace std;
using namespace cv;
using namespace flann;
//using namespace METAIO_NAMESPACE;



int _tmain(int argc, _TCHAR* argv[])

{
	cout<<"***Lets Run VTK***"<<endl;
	//Read the image
  vtkSmartPointer<vtkTIFFReader> reader =
    vtkSmartPointer<vtkTIFFReader>::New();
  reader->SetFileName ( "S:\\GSoC_2015_Implementation\\BsplineInterpolationResult_First\\2.1.bspline.tif" );

 // Convert the image to a polydata
  vtkSmartPointer<vtkImageDataGeometryFilter> imageDataGeometryFilter = 
    vtkSmartPointer<vtkImageDataGeometryFilter>::New();
  imageDataGeometryFilter->SetInputConnection(reader->GetOutputPort());
  imageDataGeometryFilter->Update();




  //Read the image
  vtkSmartPointer<vtkTIFFReader> reader1 =
    vtkSmartPointer<vtkTIFFReader>::New();
  reader1->SetFileName ( "S:\\GSoC_2015_Implementation\\BsplineInterpolationResult_First\\2.10.bspline.tif" );

 // Convert the image to a polydata
  vtkSmartPointer<vtkImageDataGeometryFilter> imageDataGeometryFilter1 = 
    vtkSmartPointer<vtkImageDataGeometryFilter>::New();
  imageDataGeometryFilter1->SetInputConnection(reader1->GetOutputPort());
  imageDataGeometryFilter1->Update();

   //Append the two meshes 
  vtkSmartPointer<vtkAppendPolyData> appendFilter =
    vtkSmartPointer<vtkAppendPolyData>::New();
  appendFilter->AddInputData(imageDataGeometryFilter->GetOutput());
  appendFilter->AddInputData(imageDataGeometryFilter1->GetOutput());
  appendFilter->Update();

  // Remove any duplicate points.
  vtkSmartPointer<vtkCleanPolyData> cleanFilter =
    vtkSmartPointer<vtkCleanPolyData>::New();
  cleanFilter->SetInputConnection(appendFilter->GetOutputPort());
  cleanFilter->Update();
 
  //Create a mapper and actor
  vtkSmartPointer<vtkPolyDataMapper> mapper =
    vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(appendFilter->GetOutputPort());
 
  vtkSmartPointer<vtkActor> actor =
    vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
 
  //Create a renderer, render window, and interactor
  vtkSmartPointer<vtkRenderer> renderer =
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow =
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
 
  //Add the actors to the scene
  renderer->AddActor(actor);
  renderer->SetBackground(.3, .2, .1); // Background color dark red
 
  //Render and interact
  renderWindow->Render();
  renderWindowInteractor->Start();

 /*
  // Visualize Ploydata
   // Create a mapper and actor
  vtkSmartPointer<vtkPolyDataMapper> mapper = 
    vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(imageDataGeometryFilter->GetOutputPort());
 
  vtkSmartPointer<vtkActor> actor = 
      vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
 
  // Visualization
  vtkSmartPointer<vtkRenderer> renderer = 
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow = 
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
 
  renderer->AddActor(actor);
  renderer->SetBackground(1,1,1); // Background color white
 
  renderWindow->Render();
  renderWindowInteractor->Start();
   // Visualize Ploydata Until here
  */

	//std::cout << "Output has " << polydata->GetNumberOfPoints() << " points." << std::endl;  
 /*
  // Visualize TIFF
  vtkSmartPointer<vtkImageViewer2> imageViewer =
    vtkSmartPointer<vtkImageViewer2>::New();
  imageViewer->SetInputConnection(reader->GetOutputPort());
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  imageViewer->SetupInteractor(renderWindowInteractor);
  imageViewer->Render();
  imageViewer->GetRenderer()->ResetCamera();
  imageViewer->Render();
 
  renderWindowInteractor->Start();
  // Visualize TIFF Until here
  */
	return 0;
}