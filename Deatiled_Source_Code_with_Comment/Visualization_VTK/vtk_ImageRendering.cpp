#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "baseFunc.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <afxwin.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctype.h>


//before VTK includes
#define vtkRenderingCore_AUTOINIT 4(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingFreeTypeOpenGL,vtkRenderingOpenGL)
#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)

#include "vtkSmartPointer.h"
#include "vtkImageReader2.h"
#include "vtkMatrix4x4.h"
#include "vtkImageReslice.h"
#include "vtkLookupTable.h"
#include "vtkImageMapToColors.h"
#include "vtkImageActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkInteractorStyleImage.h"
#include "vtkCommand.h"
#include "vtkImageData.h"
#include "vtkImageMapper3D.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#include <vtkTIFFReader.h>
#include <vtkImageDataGeometryFilter.h>
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataNormals.h"
#include <vtkImageIterator.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkDataSetMapper.h>
#include <vtkClipVolume.h>
#include "vtkDataSetTriangleFilter.h"
#include "vtkNew.h"
#include <vtkThreshold.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetMapper.h>
#include <vtkSphereSource.h>
#include <vtkActor.h>
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkProperty.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkThreshold.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkMeshQuality.h>
#include <vtkSphereSource.h>
#include <vtkTriangleFilter.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
void removeDuplicates(std::vector<double>& vec)
{

   std::sort(vec.begin(), vec.end());
   vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}
// The mouse motion callback, to turn "Slicing" on and off
class vtkImageInteractionCallback : public vtkCommand
{
public:

  static vtkImageInteractionCallback *New() {
    return new vtkImageInteractionCallback; };

  vtkImageInteractionCallback() {
    this->Slicing = 0;
    this->ImageReslice = 0;
    this->Interactor = 0; };

  void SetImageReslice(vtkImageReslice *reslice) {
    this->ImageReslice = reslice; };

  vtkImageReslice *GetImageReslice() {
    return this->ImageReslice; };

  void SetInteractor(vtkRenderWindowInteractor *interactor) {
    this->Interactor = interactor; };

  vtkRenderWindowInteractor *GetInteractor() {
    return this->Interactor; };

  virtual void Execute(vtkObject *, unsigned long event, void *)
    {
    vtkRenderWindowInteractor *interactor = this->GetInteractor();

    int lastPos[2];
    interactor->GetLastEventPosition(lastPos);
    int currPos[2];
    interactor->GetEventPosition(currPos);

    if (event == vtkCommand::LeftButtonPressEvent)
      {
      this->Slicing = 1;
      }
    else if (event == vtkCommand::LeftButtonReleaseEvent)
      {
      this->Slicing = 0;
      }
    else if (event == vtkCommand::MouseMoveEvent)
      {
      if (this->Slicing)
        {
        vtkImageReslice *reslice = this->ImageReslice;

        // Increment slice position by deltaY of mouse
        int deltaY = lastPos[1] - currPos[1];

        reslice->Update();
        double sliceSpacing = reslice->GetOutput()->GetSpacing()[2];
        vtkMatrix4x4 *matrix = reslice->GetResliceAxes();
        // move the center point that we are slicing through
        double point[4];
        double center[4];
        point[0] = 0.0;
        point[1] = 0.0;
        point[2] = sliceSpacing * deltaY;
        point[3] = 1.0;
        matrix->MultiplyPoint(point, center);
        matrix->SetElement(0, 3, center[0]);
        matrix->SetElement(1, 3, center[1]);
        matrix->SetElement(2, 3, center[2]);
        interactor->Render();
        }
      else
        {
        vtkInteractorStyle *style = vtkInteractorStyle::SafeDownCast(
          interactor->GetInteractorStyle());
        if (style)
          {
          style->OnMouseMove();
          }
        }
      }
    };

private:

  // Actions (slicing only, for now)
  int Slicing;

  // Pointer to vtkImageReslice
  vtkImageReslice *ImageReslice;

  // Pointer to the interactor
  vtkRenderWindowInteractor *Interactor;
};

// The program entry point
int _tmain(int argc, _TCHAR* argv[])
{
 
  // Start by loading some data.
   vtkSmartPointer<vtkTIFFReader> reader =
    vtkSmartPointer<vtkTIFFReader>::New();
  //reader->SetFileName ( "C:\\Users\\Neel\\Desktop\\VTK_try\\test_images\\2.1.bspline.tif" );
  //reader->SetFilePrefix("C:\\Users\\Neel\\Desktop\\VTK_try\\test_images\\");
  reader->SetFileNameSliceOffset(1);
  reader->SetFileNameSliceSpacing(1);
  reader->SetFilePattern("C:\\Users\\Neel\\Desktop\\VTK_try\\Less_series_images\\2.%d.bspline.tif");
  reader->SetDataExtent(0, 63, 0, 63, 1, 111);//125
  
  reader->SetOrientationType(3);
  reader->Update();

  vtkSmartPointer<vtkThreshold> threshold = vtkSmartPointer<vtkThreshold>::New();
  threshold->SetInputConnection(reader->GetOutputPort());
  threshold->ThresholdByUpper(1);
  threshold->Update();
  vtkUnstructuredGrid* thresholdedPolydata = threshold->GetOutput();
  std::cout << "There are " << thresholdedPolydata->GetNumberOfCells() 
            << " cells after thresholding." << std::endl;

   vtkNew<vtkDataSetTriangleFilter> tf;
   tf->SetInputConnection(threshold->GetOutputPort());
   tf->Update();

  //Visualization UnstructuredGrid after threshold
    vtkUnstructuredGrid* triangulatedPolydata = tf->GetOutput();
  std::cout << "There are " << triangulatedPolydata->GetNumberOfCells() 
            << " cells after applying TriangleFilter." << std::endl;
  
  // Create a mapper and actor
  vtkSmartPointer<vtkDataSetMapper> mapper = 
    vtkSmartPointer<vtkDataSetMapper>::New();

  mapper->SetInputData(triangulatedPolydata);//thresholdedPolydata

  vtkSmartPointer<vtkActor> actor = 
    vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
  actor->GetProperty()->SetRepresentationToWireframe();
  actor->GetProperty()->SetLineWidth(5);
 
  // Create a renderer, render window, and interactor
  vtkSmartPointer<vtkRenderer> renderer = 
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow = 
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
 
  // Add the actors to the scene
  renderer->AddActor(actor);
  //renderer->AddActor(sphereActor);
  //renderer->SetBackground(1,1,1); // Background color white
 
  // Render and interact
  renderWindow->Render();
  renderWindowInteractor->Start();
  /*
  // Create an image data
  vtkSmartPointer<vtkImageData> imageData = 
    vtkSmartPointer<vtkImageData>::New();
  //vtkImageData *imageData=0;
  imageData->AllocateScalars(VTK_DOUBLE,1);
  imageData =reader->GetOutput() ;

  int* dims = imageData->GetDimensions();
  // int dims[3]; // can't do this
 
  std::cout << "Dims: " << " x: " << dims[0] << " y: " << dims[1] << " z: " << dims[2] << std::endl;
 
  std::cout << "Number of points: " << imageData->GetNumberOfPoints() << std::endl;
  std::cout << "Number of cells: " << imageData->GetNumberOfCells() << std::endl;
  int extent[6];
  imageData->GetExtent(extent) ;

  vtkIdType numberOfPointArrays = imageData->GetPointData()->GetNumberOfArrays();
  std::cout << "Number of PointData arrays: " << numberOfPointArrays << std::endl;

  cout<< "Number of Scalar Components: "<<imageData->GetNumberOfScalarComponents()<<endl;

  for(vtkIdType i = 0; i < numberOfPointArrays; i++)
    {
    // The following two lines are equivalent
    //arrayNames.push_back(polydata->GetPointData()->GetArray(i)->GetName());
    //arrayNames.push_back(polydata->GetPointData()->GetArrayName(i));
    int dataTypeID = imageData->GetPointData()->GetArray(i)->GetDataType();
    std::cout << "Array " << i << ": " << reader->GetOutput()->GetPointData()->GetArrayName(i)
              << " (type: " << dataTypeID << ")" << std::endl;
	
    }

  /*
  // Retrieve the entries from the image data and print them to the screen
  std::vector<double> vec_vals;
  for (int z = 0; z < dims[2]; z++)
    {
    for (int y = 0; y < dims[1]; y++)
      {
      for (int x = 0; x < dims[0]; x++)
        {
        double* pixel = static_cast<double*>(imageData->GetScalarPointer());
        // do something with v

        //std::cout << pixel[0] << " ";
		vec_vals.push_back(pixel[0]);
        }
      //std::cout << std::endl;
      }
    //std::cout << std::endl;
    }
   removeDuplicates(vec_vals);
 
  double valuesRange[2];
  vtkIdType inx = 0;
  vtkDoubleArray::SafeDownCast(imageData->GetPointData()->GetArray(inx))->GetValueRange(valuesRange);
  std::cout << "valuesRange = " << valuesRange[0] << " " << valuesRange[1] << std::endl;
  //
  vtkSmartPointer<vtkClipVolume> vol =
    vtkSmartPointer<vtkClipVolume>::New();

	vol->SetInputData(imageData);
	double tryval=-5.4861292803319049e+303;
	vol->SetValue(tryval);
	vol->Update();

	vtkNew<vtkDataSetTriangleFilter> tf;
    tf->SetInputConnection(vol->GetOutputPort()); 
    tf->Update();

   vtkSmartPointer<vtkDataSetMapper> mapper =
    vtkSmartPointer<vtkDataSetMapper>::New();
  mapper->SetInputData(imageData);
 
  vtkSmartPointer<vtkActor> actor =
    vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
 
  vtkSmartPointer<vtkRenderer> renderer =
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow =
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
 
  renderer->AddActor(actor);
  renderer->SetBackground(.3, .6, .3); // Background color green
 
  renderWindow->Render();
  renderWindowInteractor->Start();


  // Retrieve the entries from the image data and print them to the screen
  vtkImageIterator<double> it(imageData, extent);
  while(!it.IsAtEnd())
    {
    double* valIt = it.BeginSpan();
    double *valEnd = it.EndSpan();
    while (valIt != valEnd)
      {
      // Increment for each component
      std::cout << "("
                << *valIt++ << ","
                << *valIt++ << ","
                << *valIt++ << ") ";
      }      
    std::cout << std::endl;
    it.NextSpan();
    }
  


  //Visualization 
  // Calculate the center of the volume
  int extent[6];
  double spacing[3];
  double origin[3];


  reader->GetOutputInformation(0)->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent);
  reader->GetOutput()->GetSpacing(spacing);
  reader->GetOutput()->GetOrigin(origin);

  double center[3];
  center[0] = origin[0] + spacing[0] * 0.5 * (extent[0] + extent[1]);
  center[1] = origin[1] + spacing[1] * 0.5 * (extent[2] + extent[3]);
  center[2] = origin[2] + spacing[2] * 0.5 * (extent[4] + extent[5]);

  // Matrices for axial, coronal, sagittal, oblique view orientations
  //static double axialElements[16] = {
  //         1, 0, 0, 0,
  //         0, 1, 0, 0,
  //         0, 0, 1, 0,
  //         0, 0, 0, 1 };

  //static double coronalElements[16] = {
  //         1, 0, 0, 0,
  //         0, 0, 1, 0,
  //         0,-1, 0, 0,
  //         0, 0, 0, 1 };

  static double sagittalElements[16] = {
           0, 0,-1, 0,
           1, 0, 0, 0,
           0,-1, 0, 0,
           0, 0, 0, 1 };

  //static double obliqueElements[16] = {
  //         1, 0, 0, 0,
  //         0, 0.866025, -0.5, 0,
  //         0, 0.5, 0.866025, 0,
  //         0, 0, 0, 1 };

  // Set the slice orientation
  vtkSmartPointer<vtkMatrix4x4> resliceAxes =
    vtkSmartPointer<vtkMatrix4x4>::New();
  resliceAxes->DeepCopy(sagittalElements);
  // Set the point through which to slice
  resliceAxes->SetElement(0, 3, center[0]);
  resliceAxes->SetElement(1, 3, center[1]);
  resliceAxes->SetElement(2, 3, center[2]);

  // Extract a slice in the desired orientation
  vtkSmartPointer<vtkImageReslice> reslice =
    vtkSmartPointer<vtkImageReslice>::New();
  reslice->SetInputConnection(reader->GetOutputPort());
  reslice->SetOutputDimensionality(2);
  reslice->SetResliceAxes(resliceAxes);
  reslice->SetInterpolationModeToLinear();

  // Create a greyscale lookup table
  vtkSmartPointer<vtkLookupTable> table =
    vtkSmartPointer<vtkLookupTable>::New();
  table->SetRange(0, 2); // image intensity range 2000
  table->SetValueRange(0.0, 1.0); // from black to white
  table->SetSaturationRange(0.0, 0.0); // no color saturation
  table->SetRampToLinear();
  table->Build();

  // Map the image through the lookup table
  vtkSmartPointer<vtkImageMapToColors> color =
    vtkSmartPointer<vtkImageMapToColors>::New();
  color->SetLookupTable(table);
  color->SetInputConnection(reslice->GetOutputPort());

  // Display the image
  vtkSmartPointer<vtkImageActor> actor =
    vtkSmartPointer<vtkImageActor>::New();
  actor->GetMapper()->SetInputConnection(color->GetOutputPort());

  vtkSmartPointer<vtkRenderer> renderer =
    vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);

  vtkSmartPointer<vtkRenderWindow> window =
    vtkSmartPointer<vtkRenderWindow>::New();
  window->AddRenderer(renderer);

  // Set up the interaction
  vtkSmartPointer<vtkInteractorStyleImage> imageStyle =
    vtkSmartPointer<vtkInteractorStyleImage>::New();
  vtkSmartPointer<vtkRenderWindowInteractor> interactor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor->SetInteractorStyle(imageStyle);
  window->SetInteractor(interactor);
  window->Render();

  vtkSmartPointer<vtkImageInteractionCallback> callback =
    vtkSmartPointer<vtkImageInteractionCallback>::New();
  callback->SetImageReslice(reslice);
  callback->SetInteractor(interactor);

  imageStyle->AddObserver(vtkCommand::MouseMoveEvent, callback);
  imageStyle->AddObserver(vtkCommand::LeftButtonPressEvent, callback);
  imageStyle->AddObserver(vtkCommand::LeftButtonReleaseEvent, callback);

  // Start interaction
  // The Start() method doesn't return until the window is closed by the user
  interactor->Start();
  //Visualization Until here
  */

  return EXIT_SUCCESS;
}
