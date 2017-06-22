#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/shape.hpp"
#include "utils/Otsu.hpp"
#include "assignments/Assignment.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/Histogram.hpp"
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include "utils/Moore.hpp"
#include <fstream>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "../build/src/Helper.h"
#include "../build/src/Marker.h"

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale, image_padded, frame_padded;
const Scalar BLACK = cv::Scalar(0, 0, 0);
const Scalar WHITE = cv::Scalar(255, 255, 255);
const Vec3b RED = cv::Vec3b(0, 0, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;
Marker* markerObject;

void doEveryting(cv::Mat image_rgb, cv::VideoWriter outputVideo)
{
	auto imageOriginal = image_rgb.clone();
	auto markerLeo = imread("C:\\Proj\\lab_ocv_template\\res\\0M.png");
	if (markerLeo.empty())
	{
		cout << "Leo marker not found";
	}
	
	auto markerVan = imread("C:\\Proj\\lab_ocv_template\\res\\1M.png");
	if (markerLeo.empty())
	{
		cout << "Leo marker not found";
	}

	auto monaImage = imread("C:\\Proj\\lab_ocv_template\\res\\0P.png");
	if (markerLeo.empty())
	{
		cout << "Mona image not found";
	}

	cv::cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);
	auto imageGrayOrig = image_grayscale;
	cv::Mat markerLeoOrig = markerLeo.clone();
	cv::Mat markerVanOrig = markerVan.clone();
	cv::cvtColor(markerLeo, markerLeo, CV_RGB2GRAY);
	cv::cvtColor(markerVan, markerVan, CV_RGB2GRAY);



	// threshold imagee for edge detection
	cv::threshold(markerLeo, markerLeo, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(markerVan, markerVan, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//imshow("marker van", markerVan);
	//imshow("marker leo", markerLeo);
	markerLeo = Moore::performErosion(markerLeo, 0, 5);
	markerVan = Moore::performErosion(markerVan, 0, 9);
	//markerLeo = markerObject->preProcessImage(markerLeo);
	//markerVan = markerObject->preProcessImage(markerVan);
	//imshow("van", markerVan);
	cv::Mat cannyLeo, cannyVan;

	cv::Canny(markerLeo, cannyLeo, 50, 50 * 3, 3);
	cv::Canny(markerVan, cannyVan, 50, 50 * 3, 3);
	vector<vector<Point>> leoContours, vanContours;
	vector<Vec4i> leoH, vanH;
	cv::findContours(markerLeo, leoContours, leoH, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	cv::findContours(markerVan, vanContours, vanH, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<Moments> muLeo(leoContours.size());
	for (int i = 0; i < leoContours.size(); i++)
	{
		muLeo[i] = moments(leoContours[i], false);
	}

	vector<Moments> muVan(vanContours.size());
	for (int i = 0; i < vanContours.size(); i++)
	{
		muVan[i] = moments(vanContours[i], false);
	}

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(leoContours.size());
	vector<vector<Point> >hull1(leoContours.size());
	std::vector<cv::Point> leoAggregated;
	for (int i = 0; i < leoContours.size(); i++)
	{
		//convexHull(Mat(leoContours[i]), hull[i], false);
		leoAggregated.insert(leoAggregated.end(), leoContours[i].begin(), leoContours[i].end());
	}
	std::vector<cv::Point> vanAggregated;
	for (int i = 0; i < vanContours.size(); i++)
	{
		vanAggregated.insert(vanAggregated.end(), vanContours[i].begin(), vanContours[i].end());
	}

	Mat drawing1 = Mat::zeros(markerLeo.size(), CV_8UC3);
	Mat drawing2 = Mat::zeros(markerVan.size(), CV_8UC3);
	//for (int i = 0; i< leoContours.size(); i++)
	//{
	//	cv::Scalar color = cv::Scalar(255, 255, 0);
	//	drawContours(drawing1, leoContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	//}
	//for (int i = 0; i< vanContours.size(); i++)
	//{
	//	cv::Scalar color = cv::Scalar(255, 255, 0);
	//	drawContours(drawing2, vanContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	//}
	std::vector<std::vector<cv::Point>> vanAggregatedContainer, leoAggregatedContainer;
	vanAggregatedContainer.push_back(vanAggregated);
	leoAggregatedContainer.push_back(leoAggregated);
	cv::Scalar color = cv::Scalar(255, 255, 0);
	drawContours(drawing2, vanAggregatedContainer, 0, color, 1, 8, vector<Vec4i>(), 0, Point());
	drawContours(drawing1, leoAggregatedContainer, 0, color, 1, 8, vector<Vec4i>(), 0, Point());
	//cv::imshow("hull leo", drawing1);
	//cv::imshow("hull van", drawing2);
	std::vector< cv::Point2f > cornersCrop, cornersMarker;
	int maxCorners = 1;
	double qualityLevel = 0.01;
	double minDistance = 20.;
	cv::Mat mask;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	cv::Mat leoCircles = markerLeo.clone();
	cv::Mat vanCircles = markerVan.clone();
	cv::goodFeaturesToTrack(markerLeo, cornersCrop, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	cv::goodFeaturesToTrack(markerVan, cornersMarker, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

	for (size_t i = 0; i < cornersCrop.size(); i++)
	{
		cv::circle(leoCircles, cornersCrop[i], 10, cv::Scalar(255.), -1);
	}

	for (size_t i = 0; i < cornersMarker.size(); i++)
	{
		cv::circle(vanCircles, cornersMarker[i], 10, cv::Scalar(255.), -1);
	}

	//cv::imshow("marker leo features", leoCircles);
	//cv::imshow("marker vanfeatures", vanCircles);

	//markerLeo = Moore::performDilation(markerLeo, 0, 2);
	//cv::GaussianBlur(markerLeo, markerLeo, Size(5, 5), 0, 0);
	//imshow("gauss", markerLeo);
	//cv::threshold(markerLeo, markerLeo, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	//cv::threshold(markerVan, markerVan, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

	// detect lines
	//imshow("wat", markerLeo);
	auto markerLeoThresh = markerLeo.clone();
	cv::Mat markerLeoCanny;
	cv::Canny(markerLeo, markerLeoCanny, 250, 150 * 3, 3);
	vector<vector<Point>> markerContours, detectedContours;
	vector<Vec4i> markerHierarchy;
	cv::findContours(markerLeoCanny, markerContours, markerHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<Point> pointVector;
	for (int i = 0; i< markerContours.size(); i++)
	{
		if (markerContours[i].size() > 100)
		{
			approxPolyDP(markerContours[i], pointVector, arcLength(Mat(markerContours[i]), true) * 0.01, true);
			if (1/*pointVector.size() == 4 && cv::isContourConvex(pointVector)*/)
			{
				detectedContours.push_back(pointVector);
			}
		}
	}
	Mat watContours = Mat::zeros(markerLeo.size(), CV_8UC3);
	for (int i = 0; i< detectedContours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 0);
		drawContours(watContours, detectedContours, i, color, 2, 8, markerHierarchy, 0, Point());
	}
	//imshow("leo contour", watContours);
	pointVector.clear();
	//cv::Mat morphedMarker;
	//cv::warpPerspective(markerLeo, morphedMarker, H, Size(markerLeo.size()));
	//imshow("spe ca merge de data asta", morphedMarker);
	//Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create(10);
	//std::vector<cv::KeyPoint> markerKeypoints;
	//detector->detect(markerLeoOrig, markerKeypoints);

	// Add results to image and save.
	//cv::Mat output;
	//cv::drawKeypoints(markerLeo, markerKeypoints, output);
	//cv::imwrite("sift_result.jpg", output);
	Mat thresholdedImage, gaussianBlurred;
	//Marker* markerObject = new Marker();

	thresholdedImage = markerObject->preProcessImage(image_rgb);
	Mat drawing = Mat::zeros(thresholdedImage.size(), CV_8UC3);
	vector<Vec4i> hierarchy;
#if 0
	Mat canny_output;
	vector<vector<Point>> contours;

	vector<vector<Point>> contoursOut;
	vector<vector<Point>> contoursAll;


	/// Detect edges using canny
	Canny(thresholdedImage, canny_output, 150, 150 * 3, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));


	for (int i = 0; i< contours.size(); i++)
	{
		if (contours[i].size() > 35)
		{
			approxPolyDP(contours[i], pointVector, arcLength(Mat(contours[i]), true) * 0.03, true);
			if (pointVector.size() == 4 && cv::isContourConvex(pointVector))
			{
				auto perimeter = cv::arcLength(pointVector, true);
				if (perimeter < 200)
					continue;
				contoursOut.push_back(pointVector);
			}
			contoursAll.push_back(pointVector);
			pointVector.clear();
		}
	}


	std::vector<std::vector<cv::Point>> tooNearCandidates;
	for (size_t i = 0; i<contoursOut.size(); i++)
	{
		auto m1 = contoursOut[i];
		//calculate the average distance of each corner to the nearest corner of the other marker candidate
		for (size_t j = i + 1; j<contoursOut.size() - 1; j++)
		{
			auto m2 = contoursOut[j];
			if(std::find_first_of(m1.begin(), m1.end(),
				m2.begin(), m2.end()) != m1.end())
			{
				contoursOut.erase(contoursOut.begin() + j);
				break;
			}
		}
	}
	//for (int i = 0; i< contoursOut.size(); i++)
	//{
	//	cv::Scalar color = cv::Scalar(255, 255, 0);
	//	drawContours(drawing, contoursOut, i, color, 2, 8, hierarchy, 0, cv::Point());
	//}
	//imshow("all contour", drawing);
	//cout << endl << "contour number: " << contoursOut.size() << endl;
#endif
	auto contoursOut = markerObject->findCandidateContours(thresholdedImage);
	for (int contourId = 0; contourId < contoursOut.size(); contourId++)
	{
		auto roi = markerObject->convertContourToRoi(contoursOut[contourId]);
		cv::Mat crop = image_rgb(roi);

		std::vector<cv::Point2f> markerCornerPoints;
		markerCornerPoints.push_back(Point2f((float)0, (float)0));
		markerCornerPoints.push_back(Point2f((float)0, (float)crop.size().height));
		markerCornerPoints.push_back(Point2f((float)crop.size().width, (float)crop.size().height));
		markerCornerPoints.push_back(Point2f((float)crop.size().width, (float)0));

		//imshow("first crop", crop);
		cv::Mat processedCrop = image_grayscale(roi);
		cv::Mat originalCrop = crop.clone();

		cv::cvtColor(crop, crop, CV_RGB2GRAY);

		auto convertedContours = markerObject->orderContourPoints(contoursOut[contourId]);
		auto H = cv::getPerspectiveTransform(convertedContours, markerCornerPoints);

		cv::Mat canonicalMarker;
		cv::warpPerspective(imageGrayOrig, canonicalMarker, H, crop.size());
		cv::Mat canonicalMarkerWithCorners;
		
		cv::warpPerspective(imageGrayOrig, canonicalMarkerWithCorners, H, cv::Size(crop.size().width - 20, crop.size().height - 20));
		//cv::resize(canonicalMarker, canonicalMarker, Size(canonicalMarker.size().width * 0.8, canonicalMarker.size().height*0.8));
		cv::Rect canonicalRoi;
		canonicalRoi.x = 15;
		canonicalRoi.y = 15;
		canonicalRoi.width = canonicalMarker.size().width - 30;
		canonicalRoi.height = canonicalMarker.size().height - 30;
		auto canonicalMarkerOriginal = canonicalMarkerWithCorners.clone();
		canonicalMarker = canonicalMarker(canonicalRoi);
		//imshow("can marker", canonicalMarker);
		std::vector<cv::Point> canMarkerPoint;
		cv::Mat cirlesMarker = canonicalMarkerOriginal.clone();
		//cv::goodFeaturesToTrack(canonicalMarkerOriginal, canMarkerPoint, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
		//for (size_t i = 0; i < canMarkerPoint.size(); i++)
		//{
		//	cv::circle(cirlesMarker, canMarkerPoint[i], 10, cv::Scalar(255.), -1);
		//}
		//imshow("bigger can marker", cirlesMarker);
		//canonicalMarker = markerObject->preProcessImage(canonicalMarker);
		//auto nimic = markerObject->preProcessImage(imageOriginal);
		//cv::GaussianBlur(canonicalMarker, canonicalMarker, cv::Size(7, 7), 0, 0);
		//imshow("w", canonicalMarker);
		//canonicalMarker = canonicalMarker(Rect(30, 30, 200, 200));
		//cv::threshold(canonicalMarker, canonicalMarker, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		//cv::GaussianBlur(canonicalMarker, canonicalMarker, Size(9, 9), BORDER_DEFAULT);
		//canonicalMarker = Moore::performDilation(canonicalMarker, 2, 2);
		//canonicalMarker = Moore::performErosion(canonicalMarker, 0, 1);
		//canonicalMarker = Moore::performDilation(canonicalMarker, 0, 5);
		//cv::imshow("crop", canonicalMarker);

		//Helper::findHomographyMatrix(originalCrop, markerLeoOrig, convertedContours, markerCornerPoints);
		//Helper::findHomographyUsingContours(originalCrop, markerLeoOrig);
		if (convertedContours.size() > 0)
		{
			//Helper::findHomographyFeatures(crop, markerLeoOrig, convertedContours, markerCornerPoints, imageOriginal, roi, outputVideo);
			markerObject->findHomographyFeatures(crop, canonicalMarker, convertedContours, imageOriginal, roi, outputVideo,leoContours, vanContours, canonicalMarkerOriginal);
			auto M = cv::findHomography(convertedContours, markerCornerPoints, RANSAC, 5.0);
			cv::warpPerspective(monaImage, monaImage, M, monaImage.size());
			//imshow("2",monaImage);
			Mat wrapedMona;

			for (int i = 0; i< contoursOut.size(); i++)
			{
				Scalar color = Scalar(255, 255, 0);
				drawContours(drawing, contoursOut, i, color, 2, 8, hierarchy, 0, Point());
			}
		}
		else
		{
			cout << "no contour before call" << endl;
		}

		//cv::imshow("contours", drawing);
	}



#if 0 
	cv::threshold(markerLeo, markerLeo, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	Mat markerDst, markerDstNorm, markerNormScaled;
	Mat cropDst, cropDstNorm, cropNormScaled;
	markerDst = Mat::zeros(markerLeo.size(), CV_32FC1);
	cropDst = Mat::zeros(crop.size(), CV_32FC1);
	/// Detector parameters
	int blockSize = 3;
	int apertureSize = 7;
	double k = 0.02;

	Mat cannyCrop;
	contours.clear();
	contoursOut.clear();
	contoursAll.clear();
	cv::threshold(crop, crop, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	Canny(crop, cannyCrop, 250, 150 * 3, 3);
	findContours(cannyCrop, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	drawing = Mat::zeros(crop.size(), CV_8U);

	for (int i = 0; i< contours.size(); i++)
	{
		if (contours[i].size() < 50)
		{
			approxPolyDP(contours[i], pointVector, arcLength(Mat(contours[i]), true) * 0.01, true);
			if (pointVector.size() == 4 && cv::isContourConvex(pointVector))
			{
				contoursOut.push_back(pointVector);
			}
			contoursAll.push_back(pointVector);
		}
	}

	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}


	//processedCrop = Moore::performDilation(processedCrop, 0, 3);
	drawing = Moore::performErosion(drawing, 0, 1);
	drawing = Moore::performDilation(drawing, 0, 1);
	drawing = Moore::performErosion(drawing, 0, 1);
	//cv::imshow("", drawing);

	/// Detecting corners
	cv::cornerHarris(canonicalMarker, markerDst, blockSize, apertureSize, k, BORDER_DEFAULT);
	cv::cornerHarris(drawing, cropDst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	cv::normalize(markerDst, markerDstNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	cv::normalize(cropDst, cropDstNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	cv::convertScaleAbs(markerDstNorm, markerNormScaled);
	cv::convertScaleAbs(cropDstNorm, cropNormScaled);
	vector<Point2f> markerPoints, roiPoints;

	/// Drawing a circle around corners
	for (int j = 0; j < markerDstNorm.rows; j++)
	{
		for (int i = 0; i < markerDstNorm.cols; i++)
		{
			if ((int)markerDstNorm.at<float>(j, i) > 170)
			{
				circle(markerNormScaled, Point(i, j), 5, Scalar(0, 255, 255), 2, 8, 0);
				roiPoints.push_back(Point(i, j));
			}
		}
	}

	for (int j = 0; j < cropDstNorm.rows; j++)
	{
		for (int i = 0; i < cropDstNorm.cols; i++)
		{
			if ((int)cropDstNorm.at<float>(j, i) > 100)
			{
				circle(cropNormScaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
				markerPoints.push_back(Point(i, j));
			}
		}
	}
	//imshow("uu", markerNormScaled);
	cout << endl;
	cout << markerPoints.size() << " roi " << roiPoints.size();
	auto minAllowedContourSize = 50 * 50;
	vector<vector<Point>> cropFeatures;
	for (auto pointVector : contoursAll)
	{
		if (pointVector.size() != 4)
		{
			continue;
		}
		if (!cv::isContourConvex(pointVector)) {
			continue;
		}

		// eliminate markers where consecutive
		// points are too close together
		float minDist = crop.size().width* crop.size().width;
		for (int i = 0; i < pointVector.size(); i++) {
			auto side = Point(points[i].x - points[(i + 1) % 4].x, points[i].y - points[(i + 1) % 4].y);
			float squaredLength = (float)side.dot(side);
			// println("minDist: " + minDist  + " squaredLength: " +squaredLength);
			minDist = min(minDist, squaredLength);
		}

		if (minDist < minAllowedContourSize) {
			continue;
		}
		cropFeatures.push_back(pointVector);
	}
#endif
	//Mat H = cv::findHomography(markerPoints, roiPoints, CV_RANSAC);
	/// Draw contours
#if 0
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(image_rgb.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 7;
	int apertureSize = 3;
	// was 0.04
	double k = 0.14;
	int thresh = 180;
	int max_thresh = 255;
	/// Detecting corners
	cv::cvtColor(drawing, drawing, CV_RGB2GRAY);
	cornerHarris(drawing, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				Scalar color = Scalar(0);
				circle(dst_norm_scaled, Point(i, j), 5, color, 2, 8, 0);
			}
		}
	}
#endif
	
}

int main(int argc, char* argv[])
{
	markerObject = new Marker();
	//cv::VideoCapture capture(argv[1]);
	//if (!capture.isOpened())
	//{
	//	throw "Could not read file";
	//}
	////Assignment::runThirdAssignment(capture);
	//// Setup output video
	//cv::VideoWriter outputVideo("output.mp4",
	//	capture.get(CV_CAP_PROP_FOURCC),
	//	capture.get(CV_CAP_PROP_FPS),
	//	cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
	//cv::Mat frame;
	//for (;;)
	//{
	//	capture >> frame;
	//	if (frame.empty())
	//	{
	//		break;
	//	}
	//	doEveryting(frame, outputVideo);
	//}
	//outputVideo.release();

	cv::VideoWriter outputVideo;
	// for image; TODO add abstraction for different types of imput
	image_rgb = imread(argv[1], 1);
	if (image_rgb.empty()) {
		cout << "Cannot read image ";
		return -1;
	}
	doEveryting(image_rgb, outputVideo);
	//
	cv::waitKey(0);
	return 0;
}
