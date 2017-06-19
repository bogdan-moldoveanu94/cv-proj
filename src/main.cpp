#include <iostream>
#include <opencv2/core.hpp>
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

	//cv::VideoCapture capture(argv[1]);
	//if (!capture.isOpened())
	//{
	//	throw "Could not read file";
	//}
	//Assignment::runThirdAssignment(capture);

	// for image; TODO add abstraction for different types of imput
	image_rgb = imread(argv[1], 1);
	if (image_rgb.empty()) {
		cout << "Cannot read image ";
		return -1;
	}
	auto imageOriginal = image_rgb.clone();
	//ifstream infile;
	//infile.open("res/0P.png");
	auto markerLeo = imread("C:\\Proj\\lab_ocv_template\\res\\0M.png");
	if(markerLeo.empty())
	{
		cout << "Leo marker not found";
		return -1;
	}

	auto monaImage = imread("C:\\Proj\\lab_ocv_template\\res\\0P.png");
	if (markerLeo.empty())
	{
		cout << "Mona image not found";
		return -1;
	}

	cv::cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);
	auto imageGrayOrig = image_grayscale;
	cv::Mat markerLeoOrig = markerLeo.clone();
	cv::cvtColor(markerLeo, markerLeo, CV_RGB2GRAY);

	std::vector<cv::Point2f> markerCornerPoints;
	markerCornerPoints.push_back(Point2f((float)0, (float)0));
	markerCornerPoints.push_back(Point2f((float)0, (float)markerLeo.size().height));
	markerCornerPoints.push_back(Point2f((float)markerLeo.size().width, (float)markerLeo.size().height));
	markerCornerPoints.push_back(Point2f((float)markerLeo.size().width, (float)0));


	//markerLeo = Moore::performDilation(markerLeo, 0, 2);
	//cv::GaussianBlur(markerLeo, markerLeo, Size(5, 5), 0, 0);
	//imshow("gauss", markerLeo);
	cv::threshold(markerLeo, markerLeo, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	//imshow("wat", markerLeo);
	auto markerLeoThresh = markerLeo;
	Canny(markerLeo, markerLeo, 250, 150 * 3, 3);
	vector<vector<Point>> markerContours, detectedContours;
	vector<Vec4i> markerHierarchy;
	findContours(markerLeo, markerContours, markerHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
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

#if 0
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(image_rgb.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 7;
	int apertureSize = 7;
	// was 0.04
	double k = 0.14;
	int thresh = 180;
	int max_thresh = 255;
	/// Detecting corners
	cornerHarris(image_grayscale, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

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
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	namedWindow("corners", CV_WINDOW_AUTOSIZE);
	imshow("corners", dst_norm_scaled);
	waitKey(0);
	return 0;
#endif
	Mat thresholdedImage, gaussianBlurred;
	image_grayscale = Moore::performErosion(image_grayscale, 0, 3);
	//cv::adaptiveThreshold(image_grayscale, thresholdedImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 35);
	//imshow("adaptive", thresholdedImage);
	cv::GaussianBlur(image_grayscale, gaussianBlurred, Size(5,5), 0, 0);
	cv::threshold(gaussianBlurred, thresholdedImage, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

	//auto histogram = Histogram::computeHistogramVector(image_grayscale);
	//auto histImage = Histogram::computeHistogramImage(image_grayscale, 256, 256, 256);
	//cv::imwrite("hist.png", histImage);
	//auto numberOfPixels = image_grayscale.total();
	//auto otsuThresholdedImage = Otsu::computeTresholdOnImage(image_grayscale, histogram, 255, numberOfPixels);
	Mat canny_output;
	vector<vector<Point>> contours;

	vector<vector<Point>> contoursOut;
	vector<vector<Point>> contoursAll;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(thresholdedImage, canny_output, 250, 150 * 3, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	for (int i=0; i< contours.size(); i++)
	{
		if(contours[i].size() > 50)
		{
			approxPolyDP(contours[i], pointVector, arcLength(Mat(contours[i]), true) * 0.01, true);
			if (pointVector.size() == 4 && cv::isContourConvex(pointVector))
			{
				contoursOut.push_back(pointVector);
			}
			contoursAll.push_back(pointVector);
		}
	}



	auto points = contoursOut[1];
	int left, right, top, bottom;
	for (int i = 0; i < points.size(); i++)
	{
		if (i == 0) // initialize corner values
		{
			left = right = points[i].x;
			top = bottom = points[i].y;
		}

		if (points[i].x < left)
			left = points[i].x;

		if (points[i].x > right)
			right = points[i].x;

		if (points[i].y < top)
			top = points[i].y;

		if (points[i].y > bottom)
			bottom = points[i].y;
	}
	std::vector<cv::Point> box_points;
	box_points.push_back(cv::Point(left, top));
	box_points.push_back(cv::Point(left, bottom));
	box_points.push_back(cv::Point(right, bottom));
	box_points.push_back(cv::Point(right, top));
	cv::RotatedRect box = cv::minAreaRect(cv::Mat(box_points));
	cv::Rect roi;
	roi.x = box.center.x - (box.size.height / 2);
	roi.y = box.center.y - (box.size.width / 2);
	roi.width = box.size.height;
	roi.height = box.size.width;
	cout << roi.x << " " << roi.y << " " << roi.width << " " << roi.height;
	cv::Mat crop = image_rgb(roi);
	cv::Mat processedCrop = image_grayscale(roi);
	cv::Mat originalCrop = crop.clone();

	cv::cvtColor(crop, crop, CV_RGB2GRAY);


	std::vector<Point2f> convertedContours;
	for (auto i = 0; i < contoursOut[0].size(); i++)
	{
		convertedContours.push_back(cv::Point2f((float)contoursOut[0][i].x, (float)contoursOut[0][i].y));
	}
	// Sort the points in anti-clockwise order
	// Trace a line between the first and second point.
	// If the third point is at the right side, then the points are anticlockwise
	cv::Point v1 = convertedContours[1] - convertedContours[0];
	cv::Point v2 = convertedContours[2] - convertedContours[0];
	double o = (v1.x * v2.y) - (v1.y * v2.x);
	if (o < 0.0) //if the third point is in the left side, then sort in anti-clockwise order
		std::swap(convertedContours[1], convertedContours[3]);
	std::reverse(convertedContours.begin(), convertedContours.end());
	auto H = cv::getPerspectiveTransform(convertedContours, markerCornerPoints);
	cv::Mat canonicalMarker;
	cv::warpPerspective(imageGrayOrig, canonicalMarker, H, Size(255,255));
	canonicalMarker = canonicalMarker(Rect(30, 30, 200, 200));
	//cv::threshold(canonicalMarker, canonicalMarker, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//cv::GaussianBlur(canonicalMarker, canonicalMarker, Size(9, 9), BORDER_DEFAULT);
	//canonicalMarker = Moore::performDilation(canonicalMarker, 2, 2);
	//canonicalMarker = Moore::performErosion(canonicalMarker, 0, 1);
	//canonicalMarker = Moore::performDilation(canonicalMarker, 0, 5);
	//cv::imshow("crop", canonicalMarker);

	//Helper::findHomographyMatrix(originalCrop, markerLeoOrig, convertedContours, markerCornerPoints);
	//Helper::findHomographyUsingContours(originalCrop, markerLeoOrig);
	Helper::findHomographyFeatures(crop, markerLeoOrig, convertedContours, markerCornerPoints, imageOriginal, roi);
	auto M = cv::findHomography(convertedContours, markerCornerPoints, RANSAC, 5.0);
	cv::warpPerspective(monaImage, monaImage, M, monaImage.size());
	//imshow("2",monaImage);
	Mat wrapedMona;
#if 0


	Ptr<cv::xfeatures2d::SiftFeatureDetector> dtor = cv::xfeatures2d::SiftFeatureDetector::create(100);
	std::vector<cv::KeyPoint> markerKey, sceneKey;
	cv::Mat markerDescriptors, sceneDescriptors;
	dtor->detectAndCompute(markerLeo, noArray(), markerKey, markerDescriptors);
	dtor->detectAndCompute(crop, noArray(), sceneKey, sceneDescriptors);

	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(markerDescriptors, sceneDescriptors, matches);


	double max_dist = 0; double min_dist = 150;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < markerDescriptors.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	for (int i = 0; i < markerDescriptors.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}

	}
	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(markerLeo, markerKey, crop, sceneKey,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	imshow("Good Matches", img_matches);
	vector<Point2f> markerH, sceneH;
	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		markerH.push_back(markerKey[good_matches[i].queryIdx].pt);
		sceneH.push_back(sceneKey[good_matches[i].trainIdx].pt);
	}

	auto H = cv::findHomography(markerH, sceneH, RANSAC, 5.0);
	Mat wrapedMona;

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(markerLeo.cols, 0);
	obj_corners[2] = cvPoint(markerLeo.cols, markerLeo.rows); obj_corners[3] = cvPoint(0, markerLeo.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(markerLeo.cols, 0), scene_corners[1] + Point2f(markerLeo.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(markerLeo.cols, 0), scene_corners[2] + Point2f(markerLeo.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(markerLeo.cols, 0), scene_corners[3] + Point2f(markerLeo.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(markerLeo.cols, 0), scene_corners[0] + Point2f(markerLeo.cols, 0), Scalar(0, 255, 0), 4);
	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);
	cv::warpPerspective(monaImage, wrapedMona, H, Size(roi.size()));
	wrapedMona.copyTo(image_rgb(roi));
	imshow("wrapped mona", image_rgb);

	crop = Moore::performErosion(crop, 0, 3);
	//crop = Moore::performDilation(crop, 0, 3);
	//cv::adaptiveThreshold(crop, crop, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 25);
	cv::GaussianBlur(crop, crop, Size(5, 5), 0, 0);
	cv::threshold(crop, crop, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//imshow("wat", crop);
	std::vector<cv::KeyPoint> cropKeyPoints;
	detector->detect(originalCrop, cropKeyPoints);


	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < min(cropKeyPoints.size(), markerKeypoints.size()); i++)
	{
		//-- Get the keypoints from the good matches
		//obj.push_back(markerKeypoints[markerKeypoints[i].queryIdx].pt);
		//scene.push_back(cropKeyPoints[cropKeyPoints[i].trainIdx].pt);
		obj.push_back(markerKeypoints[i].pt);
		scene.push_back(cropKeyPoints[i].pt);
	}
	//auto homomograpgy = cv::findHomography(obj, scene, CV_RANSAC);
	Mat markerLeoTransformed;
#endif

	for (int i = 0; i< contoursOut.size(); i++)
	{
		Scalar color = Scalar(255, 255, 0);
		drawContours(drawing, contoursOut, i, color, 2, 8, hierarchy, 0, Point());		
	}
	//cv::imshow("contours", drawing);
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
				circle(markerNormScaled, Point(i, j), 5, Scalar(0,255,255), 2, 8, 0);
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
	for(auto pointVector: contoursAll)
	{
		if(pointVector.size() != 4)
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
				circle(dst_norm_scaled, Point(i, j), 5,color, 2, 8, 0);
			}
		}
	}
#endif
	cv::waitKey(0);
	return 0;
}




