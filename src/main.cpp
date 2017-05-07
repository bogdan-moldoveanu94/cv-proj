#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "utils/Histogram.hpp"
#include "utils/Otsu.hpp"
#include "utils/Moore.hpp"
#include "utils/Hough.hpp"

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
	cv::Mat frame;

	if(!capture.isOpened())
	{
		throw "Could not read file";
	}
	auto max_intensity = 255;
	//namedWindow("video", 1);

	for(;;)
	{
		capture >> frame;
		if(frame.empty())
		{
			break;
		}
		//imshow("video", frame);
		Mat frameGrayscale;
		cvtColor(frame, frameGrayscale, CV_RGB2GRAY);

		//auto histogram = Histogram::computeHistogramVector(frameGrayscale);
		//imshow("video", frameGrayscale);
		//waitKey(1);
		//Otsu otsuObj;
		//auto numberOfPixels = frameGrayscale.total();
		//auto output = otsuObj.computeTresholdOnImage(frameGrayscale, histogram, max_intensity, numberOfPixels);
		//cv::Mat dst, cdst;

		//auto border = cv::Scalar(0);
		//Mat temp(output.rows +1, output.cols +1, CV_8U);
		//temp.setTo(border);
		//cv::copyMakeBorder(output, frame_padded, 1, 1, 1, 1, BORDER_CONSTANT, border);
		//cv::copyMakeBorder(temp, dst, 1, 1, 1, 1, BORDER_CONSTANT, border);

		//Moore moreObj;

		//auto boundaryPoints = moreObj.computeBorders(frame_padded);
		//if(!boundaryPoints.empty())
		//{
		//	for (auto it = boundaryPoints.begin(); it != boundaryPoints.end(); ++it)
		//	{
		//		temp.at<Vec3b>(*it) = RED;
		//	}
		//	imshow("detected lines", temp);
		//	waitKey(20);
		//}

		Mat dst, cdst;

		Canny(frame, dst, 150, 200, 3);
		cvtColor(dst, cdst, CV_GRAY2BGR);
		vector<Vec2f> lines;
		HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
		auto linesHough = Hough::getLines(dst, 150);
		for (auto it = linesHough.begin(); it != linesHough.end(); it++)
			{
			 //cv::line(img_res, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0, 0, 255), 2, 8);
			 cv::line(cdst, it->first, it->second, RED);
			}

		//for (size_t i = 0; i < lines.size(); i++)
		//{
		//	float rho = lines[i][0], theta = lines[i][1];
		//	Point pt1, pt2;
		//	double a = cos(theta), b = sin(theta);
		//	double x0 = a*rho, y0 = b*rho;
		//	pt1.x = cvRound(x0 + 1000 * (-b));
		//	pt1.y = cvRound(y0 + 1000 * (a));
		//	pt2.x = cvRound(x0 - 1000 * (-b));
		//	pt2.y = cvRound(y0 - 1000 * (a));
		//	line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
		//}
		imshow("detected lines", cdst);
		waitKey(100);
	}

	waitKey(0);
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
	//auto max_intensity = 255;

	Otsu otsuObj;

	// just to check that the threashold has the right value
	//auto otsuTreshold = otsuObj.computeTreshold(histogram, max_intensity, numberOfPixels);
	//std::cout << "value from my impl: " << otsuTreshold << endl;

	auto output = otsuObj.computeTresholdOnImage(image_grayscale, histogram, max_intensity, numberOfPixels);
#pragma endregion

	
#pragma region moore boundaries
	Mat copyColor;
	// pad image w/ 1 pixel black border
	auto border = cv::Scalar(0);
	cv::copyMakeBorder(output, image_padded, 1, 1, 1, 1, BORDER_CONSTANT, border);
	cv::copyMakeBorder(image_rgb, copyColor, 1, 1, 1, 1, BORDER_CONSTANT, border);


	// construct white image
	Mat image_border(Mat(image_padded.rows, image_padded.cols, CV_8U));
	image_border.setTo(255);

	cv::imwrite("output.png", output);
	Moore moreObj;

	auto boundaryPoints = moreObj.computeBorders(image_padded);
	for (auto it = boundaryPoints.begin(); it != boundaryPoints.end(); ++it)
	{
		copyColor.at<Vec3b>(*it) = RED;
	}
#pragma endregion
	//cv::imshow("image", copyColor);
	//cv::imwrite("boundaries.png", copyColor);


	cv::waitKey();

	return 0;

}





