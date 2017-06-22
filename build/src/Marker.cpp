#include "Marker.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "../../src/utils/Moore.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include "Helper.h"
#include <opencv2/calib3d.hpp>
#include <iterator>
#include <opencv2/shape/shape_distance.hpp>


#define DEBUG_MODE 0
#define EPSILON 1E-5
cv::Mat Marker::markerLeo, Marker::markerVan, Marker::vanImage, Marker::monaImage, Marker::imageColor;
std::vector<cv::Point2f> Marker::markerCornerPoints;
int i;

bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2,
	cv::Point2f &r)
{
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}
Marker::Marker()
{
	monaImage = cv::imread("C:\\Proj\\lab_ocv_template\\res\\0P.png");
	if (monaImage.empty())
	{
		throw std::runtime_error("Could not open mona image file");
	}

	vanImage = cv::imread("C:\\Proj\\lab_ocv_template\\res\\1P.png");
	if (vanImage.empty())
	{
		throw std::runtime_error("Could not open van image file");
	}

	markerLeo = cv::imread("C:\\Proj\\lab_ocv_template\\res\\0M.png");
	if (markerLeo.empty())
	{
		throw std::runtime_error("Could not open leo marker image file");
	}

	markerVan = cv::imread("C:\\Proj\\lab_ocv_template\\res\\1M.png");
	if (markerVan.empty())
	{
		throw std::runtime_error("Could not open van marker image file");
	}

	// define points in counter-clockwise order starting from top-left corner
	markerCornerPoints.push_back(cv::Point2f((float)0, (float)0));
	markerCornerPoints.push_back(cv::Point2f((float)0, (float)markerLeo.size().height));
	markerCornerPoints.push_back(cv::Point2f((float)markerLeo.size().width, (float)markerLeo.size().height));
	markerCornerPoints.push_back(cv::Point2f((float)markerLeo.size().width, (float)0));
	i = 0;
}


Marker::~Marker()
{
	delete &monaImage;
	delete &vanImage;
	delete &markerLeo;
	delete &markerVan;
}

cv::Mat Marker::preProcessImage(cv::Mat image)
{
	imageColor = image.clone();
	// convert to grayscale
	cv::cvtColor(image, imageGrayscale, CV_RGB2GRAY);

	// erode image to better detect points
	imageGrayscale = Moore::performErosion(imageGrayscale, 0, 5);

	// use gaussian blur to get rid of noise
	cv::GaussianBlur(imageGrayscale, imageGrayscale, cv::Size(7, 7), 0, 0);

	// threshold imagee for edge detection
	cv::threshold(imageGrayscale, imageGrayscale, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//cv::adaptiveThreshold(imageGrayscale, imageGrayscale, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 101, 35);
	//imshow("proc img", imageGrayscale);
	return imageGrayscale;
}

std::vector<std::vector<cv::Point>> Marker::findCandidateContours(cv::Mat image) const
{
	cv::Mat cannyOutput;
	std::vector<std::vector<cv::Point>> contours, filteredContours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> pointVector;

	// Detect edges using canny
	cv::Canny(image, cannyOutput, 150, 150 * 3, 3);

	cv::findContours(cannyOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 35)
		{
			approxPolyDP(contours[i], pointVector, arcLength(cv::Mat(contours[i]), true) * 0.03, true);
			if (pointVector.size() == 4 && cv::isContourConvex(pointVector))
			{
				auto perimeter = cv::arcLength(pointVector, true);
				if (perimeter < 200)
					continue;
				filteredContours.push_back(pointVector);
			}
			pointVector.clear();
		}
	}

	// remove contours that have similar points between them
	for (size_t i = 0; i < filteredContours.size(); i++)
	{
		auto m1 = filteredContours[i];
		for (size_t j = i + 1; j < filteredContours.size() - 1; j++)
		{
			auto m2 = filteredContours[j];
			if (std::find_first_of(m1.begin(), m1.end(),
				m2.begin(), m2.end()) != m1.end())
			{
				filteredContours.erase(filteredContours.begin() + j);
				break;
			}
		}
	}

	return filteredContours;
}

cv::Rect Marker::convertContourToRoi(std::vector<cv::Point> points) const
{
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
	roi.x = box.center.x - (box.size.height / 2) - 5;
	roi.y = box.center.y - (box.size.width / 2) - 5;
	if (roi.x < 0 || roi.y < 0)
	{
		// adjust roi in case we substracted too much
		if(roi.x < 0)
		{
			roi.x = 0;
		}
		if(roi.y < 0)
		{
			roi.y = 0;
		}
	}
	roi.width = box.size.height + 10;
	roi.height = box.size.width + 10;
	//std::cout << roi.x << " " << roi.y << " " << roi.width << " " << roi.height;
	if (!(roi.x + roi.width <= imageGrayscale.cols))
	{
		roi.x = roi.x - 30;
	}
	return roi;
}

std::vector<cv::Point2f> Marker::orderContourPoints(std::vector<cv::Point> contours) const
{
	std::vector<cv::Point2f> convertedContours;
	for (auto i = 0; i < contours.size(); i++)
	{
		convertedContours.push_back(cv::Point2f((float)contours[i].x, static_cast<float>(contours[i].y)));
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

	return convertedContours;
}


void Marker::findHomographyFeatures(cv::Mat crop, cv::Mat marker, std::vector<cv::Point2f> cropPoints, cv::Mat originalImage, cv::Rect roi, cv::VideoWriter outputVideo, std::vector<std::vector<cv::Point>> leoContour,
	std::vector<std::vector<cv::Point>> vanContour, cv::Mat leoMarker, cv::Mat vanMarker, cv::Mat canonicalMarkerOriginal)
{
	auto cropPointsInImage = cropPoints;
	cropPoints.clear();
	//imshow("crop", crop);
	int morph_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
	//cv::morphologyEx(marker, marker, cv::MORPH_CLOSE, element);
	cv::resize(marker, marker, cv::Size(256, 256));
	cv::resize(canonicalMarkerOriginal, canonicalMarkerOriginal, cv::Size(256, 256));




	// use gaussian blur to get rid of noise
	cv::GaussianBlur(marker, marker, cv::Size(5, 5), 0, 0);
	cv::GaussianBlur(canonicalMarkerOriginal, canonicalMarkerOriginal, cv::Size(5, 5), 0, 0);
	// erode image to better detect points
	//marker = Moore::performErosion(marker, 0, 1);

	marker = Moore::performErosion(marker, 0, 5);
	//marker = Moore::performDilation(marker, 0, 3);
	canonicalMarkerOriginal = Moore::performErosion(canonicalMarkerOriginal, 0, 11);
	canonicalMarkerOriginal = Moore::performDilation(canonicalMarkerOriginal, 0, 3);
	//marker = Moore::performDilation(marker, 0, 1);
	// threshold imagee for edge detection
	cv::threshold(marker, marker, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(canonicalMarkerOriginal, canonicalMarkerOriginal, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//imshow("canonical marker after processing" + std::to_string(i), marker);
	cv::Mat cannyCrop;
	cv::Mat cdst = marker.clone();
	cvtColor(cdst, cdst, CV_GRAY2BGR);
	cv::Canny(marker, cannyCrop, 50, 50 * 3, 3);
	std::vector<cv::Vec2f> lines;
	std::vector<cv::Vec2f> strongLines;
	cv::HoughLines(cannyCrop, lines, 1, CV_PI / 180, 80, 0, 0);
	if(lines.size()!= 0)
	{
		float rho0 = lines[0][0], theta0 = lines[0][1];
		strongLines.push_back(lines[0]);
		for (auto i = 1; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			if (rho > rho0 + 15 || theta > theta0 + 0.15 || rho - 15 < rho0 || theta - 0.15 < theta0)
			{
				strongLines.push_back(lines[i]);
				break;
			}

		}
	}

	
	if (strongLines.size() == 2)
	{
		std::cout << "marker detected!";
	}
	std::vector<std::vector<cv::Point2f>> linesPoints;
	for (size_t i = 0; i < strongLines.size(); i++)
	{
		float rho = strongLines[i][0], theta = strongLines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		std::vector<cv::Point2f> temp;
		temp.push_back(pt1);
		temp.push_back(pt2);
		linesPoints.push_back(temp);
		line(cdst, pt1, pt2, cv::Scalar(0, 0, 255), 3, CV_AA);
	}
	//imshow("lines" + std::to_string(i), cdst);
	//compute intersection point of lines
	cv::Point2f r;
	auto found = false;
	if(linesPoints.size() == 2)
	{
		found = intersection(linesPoints[0][0], linesPoints[0][1], linesPoints[1][0], linesPoints[1][1], r);
	}
	std::vector<std::vector<cv::Point>> cropContours;
	std::vector<cv::Vec4i> cropH;
	cv::findContours(cannyCrop, cropContours, cropH, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	cropPoints = Helper::findCornersOnCrop(crop);
	//imshow("canonical marker w/ roi", canonicalMarkerOriginal);
	std::vector< cv::Point2f > cornersCrop, cornersMarker, markerFeatures, canonicalFeatures;
	int maxCorners = 1;
	double qualityLevel = 0.01;
	double minDistance = 20.;
	cv::Mat mask;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	cv::goodFeaturesToTrack(crop, cornersCrop, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	cv::goodFeaturesToTrack(marker, markerFeatures, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	cv::goodFeaturesToTrack(canonicalMarkerOriginal, canonicalFeatures, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	for (size_t i = 0; i < markerFeatures.size(); i++)
	{
		cv::circle(crop, markerFeatures[i], 10, cv::Scalar(255.), -1);
	}
	cv::Mat wat = canonicalMarkerOriginal.clone();
	for (size_t i = 0; i < canonicalFeatures.size(); i++)
	{
		cv::circle(wat, canonicalFeatures[i], 10, cv::Scalar(255.), -1);
	}
	//cv::imshow("original crop canonical feature"+std::to_string(i), wat);
	//imshow("crop feature"+std::to_string(i), crop);

	//cv::Point2f bottomRightCorner;
	//auto distance = -1;;
	//auto bottomRightPointIndex = -1;
	//for (auto i = 0; i< cropPoints.size(); i++)
	//{
	//	auto tempDistance = cv::norm(cornersCrop[0] - cropPoints[i]);
	//	if (tempDistance > distance)
	//	{
	//		distance = tempDistance;
	//		bottomRightPointIndex = i;
	//	}
	//}
	if (found)
	{

		cv::Point2f bottomRightCorner;
		auto distance = -1;;
		auto bottomRightPointIndex = -1;
		for (auto i = 0; i < markerCornerPoints.size(); i++)
		{
			//auto tempDistance = cv::norm(canonicalFeatures[0] - markerCornerPoints[i]);
			auto tempDistance = cv::norm(r - markerCornerPoints[i]);
			if (tempDistance > distance)
			{
				distance = tempDistance;
				bottomRightPointIndex = i;
			}
		}


		auto markerNumber = -1;
		std::vector<cv::Point> cropP, leoP, vanP;
		std::vector<cv::Vec4i> hierarchy;
		cv::Mat drawing = cv::Mat::zeros(marker.size(), CV_8UC3);
		int minContourSize = 0;

		std::vector<std::vector<cv::Point>> hull1(cropContours.size());
		for (int i = 0; i < cropContours.size(); i++)
		{
			//convexHull(cv::Mat(cropContours[i]), hull1[i], false);
		}
		//for (int i = 0; i< cropContours.size(); i++)
		//{
		//	cv::Scalar color = cv::Scalar(255, 255, 0);
		//	drawContours(drawing, cropContours, i, color, 2, 8, hierarchy, 0, cv::Point());
		//}
		std::vector<std::vector<cv::Point>> cropAggregatedContainer;
		i = i + 1;
		//imshow("drawing" + std::to_string(i), drawing);
		//imshow("canonical", marker);
		//cropContours = hull1;
		for (auto i = 0; i < cropContours.size(); i++)
		{
			//std::copy(leoP.begin(), leoP.end(), std::back_inserter(leoContour[i]));
			if (cropContours[i].size() > minContourSize)
			{
				cropP.insert(cropP.end(), cropContours[i].begin(), cropContours[i].end());
			}

		}
		cropAggregatedContainer.push_back(cropP);
		cv::Scalar color = cv::Scalar(255, 255, 0);
		drawContours(drawing, cropAggregatedContainer, 0, color, 2, 8, hierarchy, 0, cv::Point());
		//cv::imshow("drawing agg"+std::to_string(i), drawing);
		for (auto i = 0; i < leoContour.size(); i++)
		{
			//std::copy(leoP.begin(), leoP.end(), std::back_inserter(leoContour[i]));
			if (leoContour[i].size() > minContourSize)
			{
				leoP.insert(leoP.end(), leoContour[i].begin(), leoContour[i].end());
			}

		}
		for (auto i = 0; i < vanContour.size(); i++)
		{
			//std::copy(vanP.begin(), vanP.end(), std::back_inserter(vanContour[i]));
			if (vanContour[i].size() > minContourSize)
			{
				vanP.insert(vanP.end(), vanContour[i].begin(), vanContour[i].end());
			}

		}
		//cv::Ptr<cv::ShapeContextDistanceExtractor> distanceExtractor = cv::createShapeContextDistanceExtractor();
		//float leo1 = distanceExtractor->computeDistance(cropP, leoP);
		//float van1 = distanceExtractor->computeDistance(cropP, vanP);
		//auto leo = cv::matchShapes(cropP, leoP, 1, 0.0);
		//auto van = cv::matchShapes(cropP, vanP, 1, 0.0);

		auto leo = cv::matchShapes(leoMarker, marker, 2, 0.0);
		auto van = cv::matchShapes(vanMarker, marker, 2, 0.0);
		//imshow("leo", leoMarker);
		//imshow("van", vanMarker);
		//leo = leo1;
		//van = van1;
		if (/*leo > 1000 || van > 1500 */ strongLines.size() != 2)
		{

		}
		else
		{

			std::cout << "leo match:" << leo << std::endl;
			std::cout << "van match: " << van << std::endl;
			if (leo < van)
			{
				std::cout << "marker leo detected" << std::endl;
				std::cout << std::endl;
				markerNumber = 0;
			}
			else
			{
				std::cout << "marker van detected" << std::endl;
				std::cout << std::endl;
				markerNumber = 1;
			}
			if (cropPoints.size() > 0)
			{
				cv::Mat wrappedImage;
				auto additional = 0;
				if (markerNumber == 0)
				{
					additional = 1;
				}
				std::rotate(cropPoints.begin(), cropPoints.begin() + bottomRightPointIndex, cropPoints.end());
				std::rotate(cropPoints.begin(), cropPoints.begin() + 2 , cropPoints.end());
				cv::Mat H = cv::findHomography(markerCornerPoints, cropPoints, CV_RANSAC, 3.0);
				if (markerNumber == 0)
				{
					cv::warpPerspective(monaImage, wrappedImage, H, crop.size());
				}
				else
				{
					cv::warpPerspective(vanImage, wrappedImage, H, crop.size());
				}


				cv::Mat cropColor = originalImage(roi);
				cv::Vec3b black = (0, 0, 0);
				for (int i = 0; i < wrappedImage.size().width; i++)
				{
					for (int j = 0; j < wrappedImage.size().height; j++)
					{
						if (wrappedImage.at<cv::Vec3b>(cv::Point(i, j)) != black)
						{
							(cropColor.at<cv::Vec3b>(cv::Point(i, j)) = wrappedImage.at<cv::Vec3b>(cv::Point(i, j)));
						}
					}
				}
				cropColor.copyTo(originalImage(roi));
				outputVideo << originalImage;
				imshow("www", originalImage);
				//cv::imshow("wat", monaImage);
			}
		}
	}
}

static double maximum(double number1, double number2, double number3) {
	return std::max(std::max(number1, number2), number3);
}
static bool almostEqual(double number1, double number2) {
	return (std::abs(number1 - number2) <= (EPSILON * maximum(1.0, std::abs(number1), std::abs(number2))));
}

static bool lineIntersection(const cv::Point2f &a1, const cv::Point2f &b1, const cv::Point2f &a2,
	const cv::Point2f &b2, cv::Point2f &intersection) {
	double A1 = b1.y - a1.y;
	double B1 = a1.x - b1.x;
	double C1 = (a1.x * A1) + (a1.y * B1);

	double A2 = b2.y - a2.y;
	double B2 = a2.x - b2.x;
	double C2 = (a2.x * A2) + (a2.y * B2);

	double det = (A1 * B2) - (A2 * B1);

	if (!almostEqual(det, 0)) {
		intersection.x = static_cast<float>(((C1 * B2) - (C2 * B1)) / (det));
		intersection.y = static_cast<float>(((C2 * A1) - (C1 * A2)) / (det));

		return true;
	}

	return false;
}


