#include <iostream>
#include <opencv2/core.hpp>
#include "utils/Otsu.hpp"
#include "assignments/Assignment.hpp"

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale, image_padded, frame_padded;
const Scalar BLACK = cv::Scalar(0, 0, 0);
const Scalar WHITE = cv::Scalar(255, 255, 255);
const Vec3b RED = cv::Vec3b(0, 0, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;

int main(int argc, char* argv[])
{

	cv::VideoCapture capture(argv[1]);
	if (!capture.isOpened())
	{
		throw "Could not read file";
	}
	Assignment::runThirdAssignment(capture);

	// for image; TODO add abstraction for different types of imput
	/*image_rgb = imread(argv[1], 1);
	if (image_rgb.empty()) {
		cout << "Cannot read image ";
		return -1;
	}*/

	return 0;
}




