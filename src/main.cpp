#include <iostream>
#include <opencv2/core.hpp>
#include "utils/Otsu.hpp"
#include "assignments/Assignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/Histogram.hpp"
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include "utils/Moore.hpp"
#include <fstream>
#include "../build/src/Helper.h"
#include "../build/src/Marker.h"


cv::Mat image_rgb, image_grayscale, image_padded, frame_padded;
const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
const cv::Scalar WHITE = cv::Scalar(255, 255, 255);
const cv::Vec3b RED = cv::Vec3b(0, 0, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;
Marker* markerObject;

enum INPUT_MODE
{
	IMAGE = 0,
	VIDEO,
	WEBCAM
};

void processFrame(cv::Mat image_rgb)
{

	cv::cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);
	auto imageGrayOrig = image_grayscale;

	auto thresholdedImage = markerObject->preProcessImage(image_rgb);
	cv::Mat drawing = cv::Mat::zeros(thresholdedImage.size(), CV_8UC3);
	std::vector<cv::Vec4i> hierarchy;

	auto contoursOut = markerObject->findCandidateContours(thresholdedImage);
	for (auto contourId = 0; contourId < contoursOut.size(); contourId++)
	{
		auto roi = markerObject->convertContourToRoi(contoursOut[contourId]);
		auto crop = image_rgb(roi);

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
		if (canonicalRoi.width < 10)
		{
			canonicalRoi.width = 10;
		}
		if (canonicalRoi.height < 10)
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
			std::cout << "no contour before call" << std::endl;
		}
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
		auto fps = capture.get(CV_CAP_PROP_FPS);
		markerObject = new Marker(fps, outputVideo);
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
		auto fps = capture.get(CV_CAP_PROP_FPS);
		markerObject = new Marker(fps, outputVideo);
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
	default: {
		std::cout << "no input mode available. exiting..." << std::endl;
		return -1;
	};
	}
}
