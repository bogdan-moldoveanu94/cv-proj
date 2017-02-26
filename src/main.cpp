#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale;
const char* keys =
{
	"{help h||}{@image |../res/lenna.png|input image name}"
};

int main( int argc, char* argv[] )
{
	CommandLineParser parser(argc, argv, keys);
	
	// first argument of the program is the image name
	string filename = parser.get<string>(0);
	cout << filename << endl;
	image_rgb = imread(filename, 1);
	if (image_rgb.empty()) {
		cout << "Cannot read image " << filename.c_str();
		return -1;
	}
	cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);

#pragma region compute histogram
	vector<int> hist(256, 0);
	for (int row = 0; row < image_grayscale.rows; ++row) {
		for (int column = 0; column < image_grayscale.cols; ++column) {
			++hist[image_grayscale.at<uchar>(row, column)];
		}
	}
	// do i really need to sort them?
	//vector<int> indices(256);
	//iota(indices.begin(), indices.end(), 0); // 0, 1, ... 255
	//sort(indices.begin(), indices.end(), [&hist](int lhs, int rhs) { return hist[lhs] > hist[rhs]; });

	// create an image to save the histogram in
	int histSize = 256;
	int hist_w = 256; 
	int hist_h = 256;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i=1; i < histSize; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
#pragma endregion


	imwrite("hist.png", histImage);
	namedWindow("image", 1);
	imshow("image", histImage);
	waitKey();

	return 0;
	
}


