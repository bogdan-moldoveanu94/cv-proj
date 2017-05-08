#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "utils/Histogram.hpp"
#include "utils/Otsu.hpp"
#include "utils/Moore.hpp"
#include "utils/Hough.hpp"
#include <fstream>

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale, image_padded, frame_padded;
const Scalar BLACK = cv::Scalar(0, 0, 0);
const Scalar WHITE = cv::Scalar(255, 255, 255);
const Vec3b RED = cv::Vec3b(0, 0, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;

double rad2degree(double rad)
{
	return rad * 180 / CV_PI;
}

double angleToVel(double angle, double rho, bool isRightSide)
{
	if(angle > 60 && angle < 90 && rho < 0 && !isRightSide)
	{
		return 4 / 3.0*(angle - 60);
	}
	else if (angle > 90 && angle < 180 && rho < 0 && !isRightSide)
	{
		return 45 + 110.0*(angle - 90) / 90;
	}
	else if( angle < 45 && rho > 0)
	{
		return 155 + 75.0*(angle / 45);
	}
	else if(angle > 45 && angle < 60 && isRightSide)
	{
		return 215 + 35.0*(angle - 45)/60;
	}
	else if(angle > 60 && isRightSide)
	{
		return 255 + 80.0* (angle - 60) /65;
	}
	else
	{
		return -1; // neither of those so it could not be determined
	}
	
	return 0;
}

int main(int argc, char* argv[])
{
	cv::VideoCapture capture(argv[1]);
	std::ofstream outfile("velocity.csv");
	cv::Mat frame;

	if(!capture.isOpened())
	{
		throw "Could not read file";
	}
	auto max_intensity = 255;
	//namedWindow("video", 1);
	cv::Rect leftDialROI(50, 120, 570, 550);
	int frames = 0;
	double prevVelocity = 0;
	auto frameNr = 0;
	//capture(leftDialROI);
	for(;;)
	{
		capture >> frame;
		if(frame.empty())
		{
			break;
		}
		//imshow("video", frame);
		frameNr++;

		cv::Mat leftDialAreaOrig = frame(leftDialROI);
		cv::Mat leftDialAreaGray;
		cv::cvtColor(leftDialAreaOrig, leftDialAreaGray, CV_RGB2GRAY);
		Mat frameGrayscale, dst, cdst;
		cv::GaussianBlur(leftDialAreaGray, dst, cv::Size(5,5), 0, 0);
		//auto erosionRes = Moore::performErosion(dst, 1, 3);
		auto dilationRes = Moore::performDilation(dst, 0, 2);
		//auto dilationRes = leftDialAreaGray.clone();
		//dilationRes = Moore::performDilation(dilationRes, 0, 3);
		//dilationRes = Moore::performDilation(dilationRes, 0, 3);
		//auto erosion2 = Moore::performErosion(dilationRes, 1, 1);
		Canny(dilationRes, dst, 150, 450, 3);
		
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		
		findContours(dst, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0));
		for (unsigned int i = 0; i < contours.size(); i++) {
			//if (hierarchy[i][3] >= 0)   //has parent, inner (hole) contour of a closed edge (looks good)
			if (cv::contourArea(contours[i]) < 250) continue;
			drawContours(dst, contours, i, Scalar(255, 0, 0), 1, 8);
		}
		//imshow("detected lines", dilationRes);
		cv::cvtColor(dst, cdst, CV_GRAY2BGR);
		vector<Vec2f> lines;
		//cv::HoughLines(dst, lines, 1, CV_PI / 180, 115, 0, 0);
		auto linesHough = Hough::getLines(dst, 115);
		auto angleAverage = 0;
		auto rhoAverage = 1; // minus wins
		//auto linesCount = 0
		for (auto it = linesHough.begin(); it != linesHough.end(); it++)
			{
			 //cv::line(img_res, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0, 0, 255), 2, 8);
			 cv::line(cdst, it->first.first, it->first.second, RED);

			 std::cout <<"Rho: " << it->second.first << " "<< "Angle: "<< it->second.second << std::endl;
			 angleAverage += it->second.second;
			 rhoAverage *= it->second.first;
			}
		if(linesHough.size() == 0)
		{
			continue;
		}
		else
		{
			angleAverage /= linesHough.size();
			auto tempVel = angleToVel(angleAverage, rhoAverage, prevVelocity > 155 ? true : false);
			prevVelocity = tempVel != -1 ? tempVel : prevVelocity;
			cout << "Velocity: " << prevVelocity << endl;
			outfile << prevVelocity << ","<< frameNr/30.0 << std::endl;
		}

		if(lines.size()==0)
		{
			//std::cout << frames++ << std::endl;
		}
		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			//std::cout << "Rho: " << rho << " " << "Theta: " << rad2degree(theta) << endl;

			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
		}
		imshow("detected lines", cdst);
		//imshow("dilation", leftDialArea);
#if 0
		Mat frameGrayscale, dst, cdst, contours;
		vector<Vec2f> lines;
		cv::cvtColor(frame, frameGrayscale, CV_RGB2GRAY);
		auto erosionRes = Moore::performErosion(frameGrayscale, 0, 3);

		auto dilationRes = Moore::performDilation(erosionRes, 0, 1);

		dilationRes = Moore::performDilation(dilationRes, 0, 1);
		dilationRes = Moore::performDilation(dilationRes, 0, 1);
		dilationRes = Moore::performDilation(dilationRes, 0, 1);
		//cv::imshow("frame", dilationRes);
		//cv::Canny(dilationRes, dst, 200, 300, 3);
		dst = dilationRes.clone();
		//imshow("dilation", dst);
		// extract contours of the canny image:
		std::vector<std::vector<cv::Point> > contoursH;
		std::vector<cv::Vec4i> hierarchyH;

		cv::findContours(dst, contoursH, hierarchyH, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
		cv::cvtColor(dst, cdst, CV_GRAY2BGR);
		cv::HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
		// draw the contours to a copy of the input image:
		cv::Mat outputH(dilationRes.rows, dilationRes.cols, CV_8U);// = dilationRes.clone();
		outputH.setTo(0);
		cv::Mat outputColor;
		cv::cvtColor(outputH, outputColor, CV_GRAY2BGR);
		for (int i = 0; i< contoursH.size(); i++)
		{
			if (cv::contourArea(contoursH[i]) < 400) continue;
			//if (hierarchyH[i][3] < 0) continue;
			cv::drawContours(outputColor, contoursH, i, cv::Scalar(0, 0, 255), 3, 8, hierarchyH, 0);
		}
		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
		}
		//imshow("dilation", outputColor);
#endif
		cv::waitKey(20);
#if 0
		auto histogram = Histogram::computeHistogramVector(frameGrayscale);
		//imshow("video", frameGrayscale);
		//waitKey(1);
		//Otsu otsuObj;
		auto numberOfPixels = frameGrayscale.total();
		auto otsuThresholdImage = Otsu::computeTresholdOnImage(frameGrayscale, histogram, max_intensity, numberOfPixels);
		cv::Mat dst, cdst;

		auto border = cv::Scalar(0);
		Mat temp(otsuThresholdImage.rows, otsuThresholdImage.cols, CV_8U);
		temp.setTo(border);
		cv::copyMakeBorder(otsuThresholdImage, frame_padded, 1, 1, 1, 1, BORDER_CONSTANT, border);
		cv::copyMakeBorder(frame, dst, 1, 1, 1, 1, BORDER_CONSTANT, border);

		auto boundaryPoints = Moore::computeBorders(frame_padded,500, 1000);
		if(!boundaryPoints.empty())
		{
			for (auto it = boundaryPoints.begin(); it != boundaryPoints.end(); ++it)
			{
				dst.at<Vec3b>(*it) = RED;
			}
			imshow("detected lines", dst);
			waitKey(20);
		}
#endif
#if 0
		Mat dst, cdst;

		Canny(frame, dst, 150, 200, 3);
		cvtColor(dst, cdst, CV_GRAY2BGR);
		vector<Vec2f> lines;
		HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
		//auto linesHough = Hough::getLines(dst, 150);
		//for (auto it = linesHough.begin(); it != linesHough.end(); it++)
		//	{
		//	 //cv::line(img_res, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0, 0, 255), 2, 8);
		//	 cv::line(cdst, it->first, it->second, RED);
		//	}

		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
		}
		imshow("detected lines", cdst);
	
		waitKey(100);
#endif
	}

	cv::waitKey(0);
	outfile.close();
	image_rgb = imread(argv[1], 1);
	if (image_rgb.empty()) {
		cout << "Cannot read image ";
		return -1;
	}
	// convert image to grayscale
	cv::cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);

#pragma region compute histogram


	vector<int> histogram = Histogram::computeHistogramVector(image_grayscale);

	auto histImage = Histogram::computeHistogramImage(image_grayscale, 256, 256, 256);
#pragma endregion


#pragma region otsu threshold computation
	auto numberOfPixels = image_grayscale.total();
	//auto max_intensity = 255;


	// just to check that the threashold has the right value
	//auto otsuTreshold = otsuObj.computeTreshold(histogram, max_intensity, numberOfPixels);
	//std::cout << "value from my impl: " << otsuTreshold << endl;

	auto output = Otsu::computeTresholdOnImage(image_grayscale, histogram, max_intensity, numberOfPixels);
#pragma endregion

	
#pragma region moore boundaries
	Mat copyColor;
	// pad image w/ 1 pixel black border
	auto borderBlack = cv::Scalar(0);
	cv::copyMakeBorder(output, image_padded, 1, 1, 1, 1, BORDER_CONSTANT, borderBlack);
	cv::copyMakeBorder(image_rgb, copyColor, 1, 1, 1, 1, BORDER_CONSTANT, borderBlack);


	// construct white image
	Mat image_border(Mat(image_padded.rows, image_padded.cols, CV_8U));
	image_border.setTo(255);

	cv::imwrite("output.png", output);


	auto boundaryPointsImg = Moore::computeBorders(image_padded,100,200);
	for (auto it = boundaryPointsImg.begin(); it != boundaryPointsImg.end(); ++it)
	{
		copyColor.at<Vec3b>(*it) = RED;
	}
#pragma endregion
	//cv::imshow("image", copyColor);
	//cv::imwrite("boundaries.png", copyColor);


	cv::waitKey();

	return 0;

}





