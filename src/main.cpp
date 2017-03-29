#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "utils/Histogram.hpp"
#include "utils/Otsu.hpp"
#include "utils/Moore.hpp"

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale, image_padded;
const Scalar BLACK = cv::Scalar(0, 0, 0);
const Scalar WHITE = cv::Scalar(255, 255, 255);
const Vec3b RED = cv::Vec3b(0, 0, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;


int main(int argc, char* argv[])
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
	auto numberOfPixels = image_grayscale.total();
	auto max_intensity = 255;

	Otsu otsuObj;

	// just to check that the threashold has the right value
	//auto otsuTreshold = otsuObj.computeTreshold(histogram, max_intensity, numberOfPixels);
	//std::cout << "value from my impl: " << otsuTreshold << endl;

	auto output = otsuObj.computeTresholdOnImage(image_grayscale, histogram, max_intensity, numberOfPixels);
#pragma endregion

	// use built-in thresholding function from opencv in order to test our implementation

	Mat otsu;
	double th_val = cv::threshold(image_grayscale, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//std::cout << "value from built in: " << th_val << endl;
	otsu = otsu > th_val;
	cv::imwrite("otsucv.png", otsu);

	Mat copyColor;
	// pad image w/ 1 pixel white border
	auto border = cv::Scalar(0);
	std::cout << "otsu image: " << "rows: " << output.rows << " cols: " << output.cols << endl;
	cv::copyMakeBorder(otsu, image_padded, 1, 1, 1, 1, BORDER_CONSTANT, border);
	cv::copyMakeBorder(image_rgb, copyColor, 1, 1, 1, 1, BORDER_CONSTANT, border);
	std::cout << "padded image: " << "rows: " << image_padded.rows << " cols: " << image_padded.cols;



	// construct white image
	Mat image_border(Mat(image_padded.rows, image_padded.cols, CV_8U));
	image_border.setTo(255);

	//cv::imwrite("hist.png", histImage);
	cv::imwrite("output.png", output);

	// comment out debugging code
	//cv::namedWindow("image", 1);
	auto output1 = Moore(image_padded, copyColor);
	//cout << output1;
	//cv::drawContours(copyColor, output1, 1, RED, 2, 8);
	//copyColor.setTo(Scalar(0, 0, 255), output1);
	copyColor += output1;
	//cv::Rect roi(0, 0, output1.size().width, output1.size().height);
	//output1.copyTo(copyColor(roi));
	//cv::imshow("image", copyColor);
	//cv::imwrite("boundaries.png", copyColor);

	//cv::imshow("image2", otsu);
	//cv::imshow("hist", histImage);
	cv::waitKey();

	return 0;

}





