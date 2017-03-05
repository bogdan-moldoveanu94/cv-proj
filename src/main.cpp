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
	vector<int> hist(image_grayscale.total());
	for (auto row = 0; row < image_grayscale.rows; row++) {
		for (auto column = 0; column < image_grayscale.cols; column++) {
			hist[ 0xFF & image_grayscale.at<uchar>(row, column)]++;
		}
	}

	vector<int> histCopy = hist;



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
	int numberOfPixels = image_grayscale.total();
	float threshold, var_max, sum, sum_B, q1, q2, miu1, miu2, sigma_t;
	threshold = var_max = sum = sum_B = q1 = q2 = miu1 = miu2 = sigma_t = 0.0;
	int max_intensity = 255;
	// Calculate histogram

	vector<int> histData(numberOfPixels);
	int ptr = 0;
	while (ptr < numberOfPixels) {
		int h = 0xFF & image_grayscale.at<uchar>(ptr);
		histData[h] ++;
		ptr++;
	}
	//int itr = 0;
	//while(itr < numberOfPixels)
	//{
	//	if(histData[itr] != hist[itr])
	//	{
	//		cout << histData[itr] << " " << hist[itr] << endl;
	//	}
	//	itr++;
	//}
	cout << hist.size() << " and the histData " << histData.size() << endl;

	// auxilary value for computing miu_2
	for (int i = 0; i <= max_intensity; i++) {
		sum += i * histCopy.at(i);
		// try w/ the new histogram
		//sum += i * histData.at(i);
	}

	// update q_i(t)
	for (int t = 0; t <= max_intensity; t++) {
		q1 += histCopy.at(t);
		// same here, try w/ new histogram
		//q1 += histData.at(t);
		if (q1 == 0) {
			continue;
		}

		q2 = numberOfPixels - q1;
		if (q2 == 0)
			break;

		// update miu_i(t)

		sum_B += t * histCopy.at(t);
		//sum_B += t * histData.at(t);
		miu1 = sum_B / q1;
		miu2 = (sum - sum_B) / q2;

		sigma_t = q1 * q2 * ((miu1 - miu2) * (miu1 - miu2));
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
	std::cout << "value from my impl: " << threshold << endl;
	Mat output = image_grayscale > static_cast<int>(threshold);
	Mat otsu;
#pragma endregion

	// use built-in thresholding function from opencv in order to test our implementation
	double th_val = cv::threshold(image_grayscale, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	std::cout << "value from built in: " << th_val << endl;
	cv::imwrite("hist.png", histImage);
	cv::imwrite("otsu.png", otsu);
	cv::imwrite("output.png", output);
	cv::namedWindow("image", 1);
	cv::imshow("image", output);
	cv::imshow("image2", otsu);
	cv::waitKey();

	return 0;
	
}


