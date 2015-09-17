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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

//#define VTK_DATA_ROOT "S:\\\VTK\\\Data\\VTKData\\\Data\\"  


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
  //reader->SetFileName ( "C:\\Users\\Neel\\Desktop\\VTK_try\\test_images\\2.1.bspline.tif" );
  //reader->SetFilePrefix("C:\\Users\\Neel\\Desktop\\VTK_try\\test_images\\");
  reader->SetFileNameSliceOffset(1);
  reader->SetFileNameSliceSpacing(1);
  reader->SetFilePattern("S:\\GSoC_2015_Implementation\\test_images\\2.%d.bspline.tif");
  reader->SetDataExtent(0, 63, 0, 63, 1, 14);
  reader->SetOrientationType(3);
  reader->Update();

 
  
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
 
	return 0;
}