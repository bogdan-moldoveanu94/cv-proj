#include "Assignment.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include "../utils/Histogram.hpp"
#include <opencv2/imgcodecs.hpp>
#include "../utils/Otsu.hpp"

#define MAX_INTENSITY 255

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
	
}
