#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "utils/Histogram.hpp"
#include "utils/Otsu.hpp"

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale;


int main( int argc, char* argv[] )
{
	image_rgb = imread(argv[1], 1);
	if (image_rgb.empty()) {
		cout << "Cannot read image ";
		return -1;
	}
	// convert image to grayscale
	cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);

#pragma region compute histogram

	Histogram histogramObj;
	vector<int> histogram = histogramObj.computeHistogramVector(image_grayscale);

	auto histImage = histogramObj.computeHistogramImage(image_grayscale, 256, 256, 256);
#pragma endregion
	

#pragma region otsu threshold computation
	int numberOfPixels = image_grayscale.total();
	auto max_intensity = 255;

	Otsu otsuObj;
	auto otsuTreshold = otsuObj.computeTreshold(histogram, max_intensity, numberOfPixels);
	std::cout << "value from my impl: " << otsuTreshold << endl;
	auto output = otsuObj.computeTresholdOnImage(image_grayscale, histogram, max_intensity, numberOfPixels);

#pragma endregion

	// use built-in thresholding function from opencv in order to test our implementation
	Mat otsu;
	double th_val = cv::threshold(image_grayscale, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	std::cout << "value from built in: " << th_val << endl;
	cv::imwrite("hist.png", histImage);
	cv::imwrite("otsu.png", otsu);
	cv::imwrite("output.png", output);
	cv::namedWindow("image", 1);
	cv::imshow("image", output);
	cv::imshow("image2", otsu);
	cv::imshow("hist", histImage);
	cv::waitKey();

	return 0;
	
}


