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
	

#pragma region otsu threshold computation
	int numberOfPixels = image_grayscale.rows * image_grayscale.cols;
	int threshold, var_max, sum, sum_B, q1, q2, miu1, miu2, sigma_t;
	threshold = var_max = sum = sum_B = q1 = q2 = miu1 = miu2 = sigma_t = 0;
	int max_intensity = 255;

	// auxilary value for computing miu_2
	for (int i = 0; i <= max_intensity; i++) {
		sum += i * hist.at(i);
	}

	// update q_i(t)
	for (int t = 0; t <= max_intensity; t++) {
		q1 += hist.at(t);
		if (q1 == 0) {
			continue;
		}
		q2 = numberOfPixels - q1;

		// update miu_i(t)

		sum_B += t * hist.at(t);
		miu1 = sum_B / q1;
		miu2 = (sum - sum_B) / q2;

		sigma_t = q1 * q2 * (miu1 - miu2) ^ 2;
		if (sigma_t > var_max) {
			threshold = t;
			var_max = sigma_t;
		}
	}
	//Mat output(image_grayscale.rows, image_grayscale.cols, CV_8UC(1), Scalar::all(0));;
	//for (int i = 0; i < numberOfPixels; i++) {
	//	//cout << (int)image_grayscale.at<uchar>(i) << " ";
	//	if ((int)image_grayscale.at<uchar>(i) > threshold) {
	//		output.at<uchar>(i) = 1;
	//	}
	//	else {
	//		output.at<uchar>(i) = 0;
	//	}
	//}

	Mat output = image_grayscale > (int)threshold;
#pragma endregion


	imwrite("hist.png", histImage);
	imwrite("output.png", output);
	namedWindow("image", 1);
	imshow("image", output);
	waitKey();

	return 0;
	
}


