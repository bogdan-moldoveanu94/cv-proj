#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/shape.hpp"
#include "utils/Otsu.hpp"
#include "assignments/Assignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/Histogram.hpp"
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include "utils/Moore.hpp"
#include <fstream>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "../build/src/Helper.h"
#include "../build/src/Marker.h"

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale, image_padded, frame_padded;
const Scalar BLACK = cv::Scalar(0, 0, 0);
const Scalar WHITE = cv::Scalar(255, 255, 255);
const Vec3b RED = cv::Vec3b(0, 0, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;
Marker* markerObject;

void doEveryting(cv::Mat image_rgb, cv::VideoWriter outputVideo, double fps)
{

	cv::cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);
	auto imageGrayOrig = image_grayscale;

	auto thresholdedImage = markerObject->preProcessImage(image_rgb);
	Mat drawing = Mat::zeros(thresholdedImage.size(), CV_8UC3);
	vector<Vec4i> hierarchy;

	auto contoursOut = markerObject->findCandidateContours(thresholdedImage);
	for (int contourId = 0; contourId < contoursOut.size(); contourId++)
	{
		auto roi = markerObject->convertContourToRoi(contoursOut[contourId]);
		cv::Mat crop = image_rgb(roi);

		cv::cvtColor(crop, crop, CV_RGB2GRAY);

		auto convertedContours = markerObject->orderContourPoints(contoursOut[contourId]);
		auto H = cv::getPerspectiveTransform(convertedContours, Marker::markerCornerPoints);

		cv::Mat canonicalMarker;
		cv::warpPerspective(imageGrayOrig, canonicalMarker, H, crop.size());
		

		cv::Rect canonicalRoi;
		canonicalRoi.x = 15;
		canonicalRoi.y = 15;
		canonicalRoi.width = canonicalMarker.size().width - 30;
		canonicalRoi.height = canonicalMarker.size().height - 30;
		if(canonicalRoi.width <10)
		{
			canonicalRoi.width = 10;
		}
		if (canonicalRoi.height <10)
		{
			canonicalRoi.height = 10;
		}
		canonicalMarker = canonicalMarker(canonicalRoi);

		if (convertedContours.size() > 0)
		{
			markerObject->findHomographyAndWriteImage(crop, canonicalMarker, roi);
		}
		else
		{
			cout << "no contour before call" << endl;
		}
	}

}

int main(int argc, char* argv[])
{

	//VideoCapture cap(0); // open the default camera
	//if (!cap.isOpened())  // check if we succeeded
	//	return -1;

	//Mat edges;
	//namedWindow("edges", 1);
	//// Setup output video
	//cv::VideoWriter outputVideo("output.avi",
	//	cap.get(CV_CAP_PROP_FOURCC),
	//	cap.get(CV_CAP_PROP_FPS),
	//	cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
	//auto fps = cap.get(CV_CAP_PROP_FPS);
	//for (;;)
	//{
	//	Mat frame;
	//	cap >> frame; // get a new frame from camera
	//	//cvtColor(frame, edges, COLOR_BGR2GRAY);
	//	//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
	//	//Canny(edges, edges, 0, 30, 3);
	//	//imshow("edges", edges);
	//	doEveryting(frame, outputVideo, fps);
	//	//if (waitKey(30) >= 0) break;
	//}
	// the camera will be deinitialized automatically in VideoCapture destructor
	//return 0;


	cv::VideoCapture capture(argv[1]);
	if (!capture.isOpened())
	{
		throw "Could not read file";
	}
	//Assignment::runThirdAssignment(capture);
	// Setup output video
	cv::VideoWriter outputVideo("output.avi",
		capture.get(CV_CAP_PROP_FOURCC),
		capture.get(CV_CAP_PROP_FPS),
		cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
	auto fps = capture.get(CV_CAP_PROP_FPS);
	markerObject = new Marker(fps, outputVideo);
	cv::Mat frame;
	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	for (;;)
	{
		capture >> frame;
		//imshow("MyVideo", frame);
		//cv::waitKey(1000 / fps);
		if (frame.empty())
		{
			break;
		}
		doEveryting(frame, outputVideo, fps);
	}
	outputVideo.release();

	//cv::VideoWriter outputVideo;
	//// for image; TODO add abstraction for different types of imput
	//image_rgb = imread(argv[1], 1);
	//if (image_rgb.empty()) {
	//	cout << "Cannot read image ";
	//	return -1;
	//}
	//doEveryting(image_rgb, outputVideo);
	
	cv::waitKey(0);
	return 0;
}
