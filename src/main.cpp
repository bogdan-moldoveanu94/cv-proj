#include <iostream>
#include <opencv2/core.hpp>
#include "utils/Otsu.hpp"
#include "assignments/Assignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include "utils/Moore.hpp"
#include <fstream>
#include "../build/src/Helper.h"
#include "../build/src/Marker.h"
#include <ctime>

#define NUM_FRAMES 60
cv::Mat image_rgb, image_grayscale, image_padded, frame_padded;
const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
const cv::Scalar WHITE = cv::Scalar(255, 255, 255);
const cv::Vec3b RED = cv::Vec3b(0, 0, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;
Marker* markerObject;
int fps = 0;
cv::VideoWriter outputVideo;
enum INPUT_MODE
{
	IMAGE = 0,
	VIDEO,
	WEBCAM
};

void processFrame(cv::Mat image)
{

	cv::cvtColor(image, image_grayscale, CV_RGB2GRAY);
	auto imageGrayOrig = image_grayscale;

	auto thresholdedImage = markerObject->preProcessImage(image);
	cv::Mat drawing = cv::Mat::zeros(thresholdedImage.size(), CV_8UC3);
	std::vector<cv::Vec4i> hierarchy;

	auto contoursOut = markerObject->findCandidateContours(thresholdedImage);
	for (auto contourId = 0; contourId < contoursOut.size(); contourId++)
	{
		auto roi = markerObject->convertContourToRoi(contoursOut[contourId]);
		auto crop = image(roi);

		cv::cvtColor(crop, crop, CV_RGB2GRAY);

		auto convertedContours = markerObject->orderContourPoints(contoursOut[contourId]);
		auto H = cv::getPerspectiveTransform(convertedContours, Marker::markerCornerPoints);

		cv::Mat canonicalMarker;
		//cv::warpPerspective(imageGrayOrig, canonicalMarker, H, crop.size());
		cv::warpPerspective(image, canonicalMarker, H, cv::Size(256,256));

		cv::Rect canonicalRoi;
		canonicalRoi.x = 18;
		canonicalRoi.y = 18;
		canonicalRoi.width = canonicalMarker.size().width -30;
		canonicalRoi.height = canonicalMarker.size().height - 30;
		if (canonicalRoi.width < 15)
		{
			canonicalRoi.width = 15;
		}
		if (canonicalRoi.height < 15)
		{
			canonicalRoi.height = 15;
		}
		canonicalMarker = canonicalMarker(canonicalRoi);

		if (convertedContours.size() > 0)
		{
			markerObject->findHomographyAndWriteImage(crop, canonicalMarker, roi);
		}
		else
		{
#if DEBUG_MODE 
			std::cout << "no contour before call" << std::endl;
#endif
		}
	}
	cv::imshow("output", Marker::imageColor);
	if (fps != 0)
	{
		cv::waitKey(1000 / fps);
		outputVideo << Marker::imageColor;
	}
}

int main(int argc, char* argv[])
{

	switch (atoi(argv[2]))
	{
	case IMAGE:
	{
		markerObject = new Marker();
		auto image = cv::imread(argv[1], 1);
		if (image.empty()) {
			std::cout << "Cannot read image ";
			return -1;
		}
		processFrame(image);
		cv::waitKey(0);
		return 0;
	};
	case VIDEO:
	{
		cv::VideoCapture capture(argv[1]);
		if (!capture.isOpened())
		{
			throw "Could not read video file";
		}
		// Setup output video
		cv::VideoWriter outputVideo("output.avi",
			capture.get(CV_CAP_PROP_FOURCC),
			capture.get(CV_CAP_PROP_FPS),
			cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
		fps = capture.get(CV_CAP_PROP_FPS);
		markerObject = new Marker();
		cv::Mat frame;
		for (;;)
		{
			capture >> frame;
			if (frame.empty())
			{
				break;
			}
			processFrame(frame);
		}
		outputVideo.release();
		return 0;
	};
	case WEBCAM:
	{
		cv::VideoCapture capture(0);
		if (!capture.isOpened())
		{
			throw "Could not read video file";
		}
		// Setup output video
		cv::VideoWriter outputVideo("output.avi",
			capture.get(CV_CAP_PROP_FOURCC),
			capture.get(CV_CAP_PROP_FPS),
			cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT)));

		markerObject = new Marker();
		// Start and end times
		time_t start, end;
		cv::Mat frame;
		// Start time
		time(&start);
		for(auto i = 0;i<NUM_FRAMES; i++)
		{
			capture >> frame;
		}
		// End Time
		time(&end);
		// Time elapsed
		double seconds = difftime(end, start);
		// Calculate frames per second
		fps = NUM_FRAMES / seconds;
		for (;;)
		{
			capture >> frame;
			if (frame.empty())
			{
				break;
			}
			processFrame(frame);
		}
		outputVideo.release();
		return 0;
	};
	default: {
		std::cout << "no input mode available. exiting..." << std::endl;
		return -1;
	};
	}
}
