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
#include <vtkImageDataGeometryFilter.h>
#include <vtkTIFFReader.h>


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

	//
	// This example reads a volume dataset, extracts an isosurface that
	// represents the skin and displays it.
	//

	// Create the renderer, the render window, and the interactor. The renderer
	// draws into the render window, the interactor enables mouse- and
	// keyboard-based interaction with the scene.
	//
	vtkRenderer *aRenderer = vtkRenderer::New();
	vtkRenderWindow *renWin = vtkRenderWindow::New();
	renWin->AddRenderer(aRenderer);
	vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
	iren->SetRenderWindow(renWin);

	// The following reader is used to read a series of 2D slices (images)
	// that compose the volume. The slice dimensions are set, and the
	// pixel spacing. The data Endianness must also be specified. The reader
	// usese the FilePrefix in combination with the slice number to construct
	// filenames using the format FilePrefix.%d. (In this case the FilePrefix
	// is the root name of the file: quarter.)
	/*
	vtkVolume16Reader *v16 = vtkVolume16Reader::New();
	v16->SetDataDimensions(64, 64);
	v16->SetDataByteOrderToLittleEndian();
	//v16->SetFilePrefix( VTK_DATA_ROOT "\\headsq\\quarter");
	//v16->SetImageRange(1, 93);
	//v16->SetDataSpacing( 3.2, 3.2, 1.5);
	v16->SetFilePrefix("S:\\GSoC_2015_Implementation\\temp1\\Image");	//"S:\\GSoC_2015_Implementation\\Registered_images_entire\\output_image"
	v16->SetImageRange(1, 551);
	v16->SetDataSpacing( 3.2, 3.2, 1.5);
	v16->SetDataOrigin(0, 0, 0);
	v16->Update();
	*/
	
    
	//Read using vtkImageReader2

	vtkSmartPointer<vtkTIFFReader> v16 =
    vtkSmartPointer<vtkTIFFReader>::New();
	v16->SetFileNameSliceOffset(1);
	v16->SetFileNameSliceSpacing(1);
	//v16->SetFilePrefix("S:\\GSoC_2015_Implementation\\temp1\\Image");
	//v16->SetDataExtent(0, 599, 0, 599, 1, 551);
	v16->SetFilePattern("S:\\GSoC_2015_Implementation\\test_images\\2.%d.bspline.tif");
	v16->SetDataOrigin(0.0, 0.0, 0.0);
	v16->SetDataExtent(0, 63, 0, 63, 1, 101);
	v16->SetOrientationType(3);
	v16->Update();
	
	// An isosurface, or contour value of 500 is known to correspond to the
	// skin of the patient. Once generated, a vtkPolyDataNormals filter is
	// is used to create normals for smooth surface shading during rendering.
	// The triangle stripper is used to create triangle strips from the
	// isosurface these render much faster on may systems.
	vtkContourFilter *skinExtractor = vtkContourFilter::New();
	skinExtractor->SetInputConnection(v16->GetOutputPort());
	skinExtractor->SetValue(0,500);
	vtkPolyDataNormals *skinNormals = vtkPolyDataNormals::New();
	skinNormals->SetInputConnection(skinExtractor->GetOutputPort());
	skinNormals->SetFeatureAngle(60.0);
	vtkPolyDataMapper *skinMapper = vtkPolyDataMapper::New();
	skinMapper->SetInputConnection(skinNormals->GetOutputPort());
	skinMapper->ScalarVisibilityOff();
	vtkActor *skin = vtkActor::New();
	skin->SetMapper(skinMapper);

	// An outline provides context around the data.
	//
	vtkOutlineFilter *outlineData = vtkOutlineFilter::New();
	outlineData->SetInputConnection(v16->GetOutputPort());
	vtkPolyDataMapper *mapOutline = vtkPolyDataMapper::New();
	mapOutline->SetInputConnection(outlineData->GetOutputPort());
	vtkActor *outline = vtkActor::New();
	outline->SetMapper(mapOutline);
	outline->GetProperty()->SetColor(1, 1, 1);

	// It is convenient to create an initial view of the data. The FocalPoint
	// and Position form a vector direction. Later on (ResetCamera() method)
	// this vector is used to position the camera to look at the data in
	// this direction.
	vtkCamera *aCamera = vtkCamera::New();
	aCamera->SetViewUp( 0, 0, -1);
	aCamera->SetPosition( 0, 1, 0);
	aCamera->SetFocalPoint( 0, 0, 0);
	aCamera->ComputeViewPlaneNormal();

	// Actors are added to the renderer. An initial camera view is created.
	// The Dolly() method moves the camera towards the FocalPoint,
	// thereby enlarging the image.
	aRenderer->AddActor(outline);
	aRenderer->AddActor(skin);
	aRenderer->SetActiveCamera(aCamera);
	aRenderer->ResetCamera();
	aCamera->Dolly(1.5);

	// Set a background color for the renderer and set the size of the
	// render window (expressed in pixels).
	aRenderer->SetBackground(0, 0, 0);
	renWin->SetSize(640, 480);

	// Note that when camera movement occurs (as it does in the Dolly()
	// method), the clipping planes often need adjusting. Clipping planes
	// consist of two planes: near and far along the view direction. The
	// near plane clips out objects in front of the plane the far plane
	// clips out objects behind the plane. This way only what is drawn
	// between the planes is actually rendered.
	aRenderer->ResetCameraClippingRange();



	iren->Initialize();
	iren->Start();

	//Prelim VTK emxample of Cylinder
	/*
	  // This creates a polygonal cylinder model with eight circumferential facets.
	  //
	  vtkCylinderSource *cylinder = vtkCylinderSource::New();
	  cylinder->SetResolution(8);
 
	  // The mapper is responsible for pushing the geometry into the graphics
	  // library. It may also do color mapping, if scalars or other attributes
	  // are defined.
	  //
	  //Exception override using tips: http://stackoverflow.com/questions/18642155/no-override-found-for-vtkpolydatamapper
	  vtkPolyDataMapper *cylinderMapper = vtkPolyDataMapper::New();//**********Exception throwing here
	  cylinderMapper->SetInputConnection(cylinder->GetOutputPort());
 
	  // The actor is a grouping mechanism: besides the geometry (mapper), it
	  // also has a property, transformation matrix, and/or texture map.
	  // Here we set its color and rotate it -22.5 degrees.
	  vtkActor *cylinderActor = vtkActor::New();
	  cylinderActor->SetMapper(cylinderMapper);
	  cylinderActor->GetProperty()->SetColor(1.0000, 0.3882, 0.2784);
	  cylinderActor->RotateX(30.0);
	  cylinderActor->RotateY(-45.0);
 
	  // Create the graphics structure. The renderer renders into the
	  // render window. The render window interactor captures mouse events
	  // and will perform appropriate camera or actor manipulation
	  // depending on the nature of the events.
	  //
	  vtkRenderer *ren1 = vtkRenderer::New();
	  vtkRenderWindow *renWin = vtkRenderWindow::New();
	  renWin->AddRenderer(ren1);
	  vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
	  iren->SetRenderWindow(renWin);
 
	  // Add the actors to the renderer, set the background and size
	  //
	  ren1->AddActor(cylinderActor);
	  ren1->SetBackground(0.1, 0.2, 0.4);
	  renWin->SetSize(200, 200);
 
	  // We'll zoom in a little by accessing the camera and invoking a "Zoom"
	  // method on it.
	  ren1->ResetCamera();
	  ren1->GetActiveCamera()->Zoom(1.5);
	  renWin->Render();
 
	  // This starts the event loop and as a side effect causes an initial render.
	  iren->Start();
 
	  // Exiting from here, we have to delete all the instances that
	  // have been created.
	  cylinder->Delete();
	  cylinderMapper->Delete();
	  cylinderActor->Delete();
	  ren1->Delete();
	  renWin->Delete();
	  iren->Delete();
    */
	

	return 0;
}