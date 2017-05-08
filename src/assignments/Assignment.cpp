#include "Assignment.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include "../utils/Histogram.hpp"
#include <opencv2/imgcodecs.hpp>
#include "../utils/Otsu.hpp"
#include "../utils/Moore.hpp"
#include <opencv2/videoio.hpp>
#include <fstream>
#include "../utils/Hough.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

const int MAX_INTENSITY = 255;
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;
const double VIDEO_FPS = 30.0;
const cv::Vec3b RED = cv::Vec3b(0, 0, 255);

void Assignment::runFirstAssignment(cv::Mat img)
{
	cv::Mat imageGrayscale;
	cv::cvtColor(img, imageGrayscale, CV_RGB2GRAY);
	auto histogram = Histogram::computeHistogramVector(imageGrayscale);
	auto histImage = Histogram::computeHistogramImage(imageGrayscale, 256, 256, 256);
	cv::imwrite("hist.png", histImage);
	auto numberOfPixels = imageGrayscale.total();
	auto otsuThresholdedImage = Otsu::computeTresholdOnImage(imageGrayscale, histogram, MAX_INTENSITY, numberOfPixels);
	cv::imwrite("output.png", otsuThresholdedImage);
}

void Assignment::runSecondAssignment(cv::Mat img)
{
	cv::Mat imageGrayscale, imageGrayscalePadded, imgCopy;
	cv::cvtColor(img, imageGrayscale, CV_RGB2GRAY);
	auto histogram = Histogram::computeHistogramVector(imageGrayscale);
	auto numberOfPixels = imageGrayscale.total();
	auto otsuThresholdedImage = Otsu::computeTresholdOnImage(imageGrayscale, histogram, MAX_INTENSITY, numberOfPixels);
	// pad image w/ 1 pixel black border
	// TODO move this to Moore class since it always needs to be done
	auto borderBlack = cv::Scalar(0);
	cv::copyMakeBorder(otsuThresholdedImage, imageGrayscalePadded, 1, 1, 1, 1, cv::BORDER_CONSTANT, borderBlack);
	cv::copyMakeBorder(img, imgCopy, 1, 1, 1, 1, cv::BORDER_CONSTANT, borderBlack);


	// construct white image
	cv::Mat image_border(cv::Mat(imageGrayscalePadded.rows, imageGrayscalePadded.cols, CV_8U));
	image_border.setTo(255);
	// comoute boundaries between 100 and 200 pixels
	auto boundaryPointsImg = Moore::computeBorders(imageGrayscalePadded, MIN_COMPONENT_LENGTH, MAX_COMPONENT_LENGTH);
	for (auto it = boundaryPointsImg.begin(); it != boundaryPointsImg.end(); ++it)
	{
		imgCopy.at<cv::Vec3b>(*it) = RED;
	}
	cv::imwrite("boundaries.png", imgCopy);
}

void Assignment::runThirdAssignment(cv::VideoCapture capture)
{
	cv::Mat frame, leftDialAreaGray, dst, cdst;
	// create csv file for storing velocity/time pairs
	std::ofstream outfile("velocity.csv");

	// define ROI of image; namely isolate pixels from left dial
	cv::Rect leftDialROI(50, 120, 570, 550);

	// count the frames in order to compute time
	auto frameNumber = 0;

	// save previously computed velocity in order to detect when line has passed the 180 degree mark
	// so we know we have speeds over 170 km/h
	double prevVelocity = 0;

	// iterate until video is over
	for(;;)
	{
		capture >> frame;
		if (frame.empty())
		{
			break;
		}
		frameNumber++;
		// crop ROI from frame
		cv::Mat leftDialAreaOrig = frame(leftDialROI);
		// convert to grayscale for canny edge detector
		cv::cvtColor(leftDialAreaOrig, leftDialAreaGray, CV_RGB2GRAY);

		// apply gaussian blur w/ kernel size of 5x5 for obtaining better results w/ canny
		cv::GaussianBlur(leftDialAreaGray, dst, cv::Size(5, 5), 0, 0);

		// leave dilation and eroisons for now until we find better aproach
		auto dilationRes = Moore::performDilation(dst, 0, 2);

		// obtain edges
		Canny(dilationRes, dst, 150, 450, 3);

		cv::cvtColor(dst, cdst, CV_GRAY2BGR);
		std::vector<cv::Vec2f> lines;

		// set threshold to 115 pixel since this seems to give the best results
		auto linesHough = Hough::getLines(dst, 115);

		// when multiple lines are produced take the average angle and the sign of r to be the product of them
		// for the velocity converter function
		auto angleAverage = 0;
		auto rhoAverage = 1; 

		for (auto it = linesHough.begin(); it != linesHough.end(); ++it)
		{
			cv::line(cdst, it->first.first, it->first.second, RED);
			//std::cout << "Rho: " << it->second.first << " " << "Angle: " << it->second.second << std::endl;
			angleAverage += it->second.second;
			rhoAverage *= it->second.first;
		}
		if (linesHough.size() == 0)
		{
			continue;
		}
		angleAverage /= linesHough.size();
		auto tempVel = angleToVelocity(angleAverage, rhoAverage, prevVelocity > 155 ? true : false);
		prevVelocity = tempVel != -1 ? tempVel : prevVelocity;
		std::cout << "Frame: " << frame << " Time elapsed: " << frameNumber / VIDEO_FPS << " Angle: " << angleAverage << " Velocity: " << prevVelocity << std::endl;
		outfile << prevVelocity << "," << frameNumber / VIDEO_FPS << std::endl;
		imshow("detected lines", cdst);
		cvWaitKey(10);
	}
	outfile.close();
	cvWaitKey(0);
}

double Assignment::angleToVelocity(double angle, double rho, bool isRightSide)
{
	if (angle > 60 && angle < 90 && rho < 0 && !isRightSide)
	{
		return 4 / 3.0*(angle - 60);
	}
	if (angle > 90 && angle < 180 && rho < 0 && !isRightSide)
	{
		return 45 + 110.0*(angle - 90) / 90;
	}
	if (angle < 45 && rho > 0)
	{
		return 155 + 75.0*(angle / 45);
	}
	if (angle > 45 && angle < 60 && isRightSide)
	{
		return 215 + 35.0*(angle - 45) / 60;
	}
	if (angle > 60 && isRightSide)
	{
		return 255 + 80.0* (angle - 60) / 65;
	}
	return -1; // neither of those so it could not be determined
}
