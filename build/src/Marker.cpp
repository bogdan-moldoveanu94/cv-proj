#include "Marker.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "../../src/utils/Moore.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include "Helper.h"
#include <opencv2/calib3d.hpp>
#include <iterator>
#include <iostream>


#define EPSILON 1E-5
#define MAX_CORNERS 15
#define MIN_EUCLIDEAN_DISTANCE 20.
#define HARRIS_FREE_PARAMETER 0.04
#define CORNER_QUALITY_LEVEL 0.01
#define BLOCK_SIZE 3
#define LEO_IMAGE 0
#define VAN_IMAGE 1
cv::Mat Marker::markerLeo, Marker::markerVan, Marker::vanImage, Marker::monaImage, Marker::imageColor;
std::vector<cv::Point2f> Marker::markerCornerPoints;

bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2,
	cv::Point2f &r)
{
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < EPSILON)
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
	preProcessMarkers();
}

void Marker::preProcessMarkers()
{
	cv::cvtColor(markerLeo, markerLeo, CV_RGB2GRAY);
	cv::cvtColor(markerVan, markerVan, CV_RGB2GRAY);

	// threshold imagee for edge detection
	cv::threshold(markerLeo, markerLeo, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(markerVan, markerVan, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	markerLeo = Moore::performErosion(markerLeo, 0, 5);
	markerVan = Moore::performErosion(markerVan, 0, 9);
}

Marker::~Marker()
{

}

cv::Mat Marker::preProcessImage(cv::Mat image)
{
	imageColor = image.clone();
	// convert to grayscale
	cv::cvtColor(image, imageGrayscale, CV_RGB2GRAY);

	// erode image to better detect points
	imageGrayscale = Moore::performErosion(imageGrayscale, 0, 5);

	// use gaussian blur to get rid of noise
	cv::GaussianBlur(imageGrayscale, imageGrayscale, cv::Size(9, 9), 0, 0);

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
	// use threshold2 as threshold1*3 as it is usually recomended
	cv::Canny(image, cannyOutput, 150, 150 * 3, 3);

	cv::findContours(cannyOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	for (int i = 0; i < contours.size(); i++)
	{
		// set an arbitray length for contours so we filter out noise
		if (contours[i].size() > 35)
		{
			// approximate contour as a polygon
			approxPolyDP(contours[i], pointVector, arcLength(cv::Mat(contours[i]), true) * 0.03, true);
			
			// keep only those that can be a square and are convex
			if (pointVector.size() == 4 && cv::isContourConvex(pointVector))
			{
				// check for perimeter length so we don't detect candidates that are too small;
				auto perimeter = cv::arcLength(pointVector, true);
				if (perimeter < 200)
					break;
				filteredContours.push_back(pointVector);
			}
			pointVector.clear();
		}
	}

	// remove contours that have similar points between them
	// so we do not do an unnecesary computation on another contour that is
	// virtually idenical with another one
	for (auto i = 0; i < filteredContours.size(); i++)
	{
		auto m1 = filteredContours[i];
		for (auto j = i + 1; j < filteredContours.size() - 1; j++)
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
	auto left = 0, right = 0, top = 0, bottom = 0;
	for (auto i = 0; i < points.size(); i++)
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
	auto box = cv::minAreaRect(cv::Mat(box_points));

	cv::Rect roi;
	// substract a little from x and y so we skim the surrounding area from the marker
	// this helps a lot during corner de
	roi.x = box.center.x - (box.size.height / 2) - 5;
	roi.y = box.center.y - (box.size.width / 2) - 5;

	// adjust roi in case we substracted too much and we get an ill formed shape and runtime error
	if (roi.x < 0 || roi.y < 0)
	{
		if (roi.x < 0)
		{
			roi.x = 0;
		}
		if (roi.y < 0)
		{
			roi.y = 0;
		}
	}
	roi.width = box.size.height + 10;
	roi.height = box.size.width + 10;
	if (!(roi.x + roi.width <= imageGrayscale.cols))
	{
		roi.x = roi.x - 10;
	}
	return roi;
}

std::vector<cv::Point2f> Marker::orderContourPoints(std::vector<cv::Point> contours) const
{
	std::vector<cv::Point2f> convertedContours;
	for (auto i = 0; i < contours.size(); i++)
	{
		convertedContours.push_back(cv::Point2f(static_cast<float>(contours[i].x), static_cast<float>(contours[i].y)));
	}
	// Sort the points in anti-clockwise order
	// Trace a line between the first and second point.
	// If the third point is at the right side, then the points are anticlockwise
	cv::Point v1 = convertedContours[1] - convertedContours[0];
	cv::Point v2 = convertedContours[2] - convertedContours[0];
	double o = (v1.x * v2.y) - (v1.y * v2.x);
	if (o < 0.0) {
		//if the third point is in the left side, then sort in anti-clockwise order
		std::swap(convertedContours[1], convertedContours[3]);
	}
	// reverse vector
	std::reverse(convertedContours.begin(), convertedContours.end());

	return convertedContours;
}

int Marker::computerOrientationFromLinePoints(cv::Mat image, std::vector<std::vector<cv::Point2f>> points) const
{
	std::vector<std::vector<cv::Point2f>> linesPoints;
	auto foundEnoughLines = Marker::detectStrongLinePoints(image, &linesPoints);
	cv::Point2f r;
	auto found = false;
	// if we do not have exactly two lines for the top left corner reject marker
	if (linesPoints.size() == 2)
	{
		// compute intersection point of the two lines
		found = intersection(linesPoints[0][0], linesPoints[0][1], linesPoints[1][0], linesPoints[1][1], r);
	}

	if (found)
	{
		// compute the location of our intersection point relative to the marker points
		// the one closest corresponds to (0, 0) in the matrix; since we have them sorted in anti-clockwise order
		// we can now know how much the points need to be shifted to be in canonical form
		cv::Point2f bottomRightCorner;
		auto distance = -1;;
		auto bottomRightPointIndex = -1;
		for (auto i = 0; i < markerCornerPoints.size(); i++)
		{
			auto tempDistance = cv::norm(r - markerCornerPoints[i]);
			if (tempDistance > distance)
			{
				distance = tempDistance;
				bottomRightPointIndex = i;
			}
		}
		return bottomRightPointIndex;
	}
	return -1;
}
bool Marker::detectStrongLinePoints(cv::Mat image, std::vector<std::vector<cv::Point2f>>* points)
{
	cv::Mat imageCanny;
	std::vector<cv::Vec2f> lines;
	std::vector<cv::Vec2f> strongLines;

	// use higher threshold so we do not detect false lines
	// second threshold is as before threshold1*3 as recomended
	cv::Canny(image, imageCanny, 150, 150 * 3, 3);

	// use 1 degree resolution; set threshold to 60 so we detect only stronger lines(i.e. our marker's edges)
	cv::HoughLines(imageCanny, lines, 1, CV_PI / 180, 60, 0, 0);
	if (lines.size() != 0)
	{
		float rho0 = lines[0][0], theta0 = lines[0][1];
		strongLines.push_back(lines[0]);
		for (auto i = 1; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			// check if the new line serves has rho which is different with more than 15 pixels from the previous line
			// also check if theta is sufficiently different(approx 10-15 degree difference)
			if (rho > rho0 + 15 || theta > theta0 + 0.3 || rho - 15 < rho0 || theta < theta0 - 0.3)
			{
				strongLines.push_back(lines[i]);
				break;
			}
		}
	}
	// we are expecting to detect two lines representing the left and top marker lines 
	// considering a canonical representation
	if (strongLines.size() != 2)
	{
		return false;
	}
	// now compute points from polar coorinates to spatial ones
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
		points->push_back(temp);
	}
	return true;
}

int Marker::detectMarkerOrientation(cv::Mat image) const
{

	auto width = image.size().width, height = image.size().height;
	std::vector< cv::Point2f > corners;

	cv::Mat mask;

	cv::goodFeaturesToTrack(image, corners, MAX_CORNERS, CORNER_QUALITY_LEVEL, MIN_EUCLIDEAN_DISTANCE, mask, BLOCK_SIZE, false, HARRIS_FREE_PARAMETER);

	// detect in which half of the image we have the most corners
	// this way we know how to shift the points for the marker orientation
	// they are arranged in anti clockwise order starting from the bottom half based on how many
	// points we need to rotate the point vector
	std::vector<int> halves(4);
	for (int x = 0; x < 4; ++x)
	{
		halves[x] = x;
	}
	for (auto i = 0; i < corners.size(); i++)
	{
		if (corners[i].y > height / 2)
		{
			// bottom half
			halves[0]++;
		}
		if (corners[i].x > width / 2)
		{
			// right half
			halves[1]++;
		}
		if (corners[i].y < height / 2)
		{
			// top half
			halves[2]++;
		}
		if (corners[i].x < width / 2)
		{
			// left half
			halves[3]++;
		}
	}
	// return the position in the vector where the most points lie
	return std::distance(halves.begin(), std::max_element(halves.begin(), halves.end()));
}

void Marker::wrapMarkerOnImage(int markerNumber, cv::Rect roi, std::vector<cv::Point2f> cropPoints, int bottomRightPointIndex) const
{
	cv::Mat wrappedImage;
	std::rotate(cropPoints.begin(), cropPoints.begin() + bottomRightPointIndex, cropPoints.end());

	auto H = cv::findHomography(markerCornerPoints, cropPoints, CV_RANSAC, 3.0);
	if (markerNumber == LEO_IMAGE)
	{
		cv::warpPerspective(monaImage, wrappedImage, H, roi.size());
	}
	else
	{
		cv::warpPerspective(vanImage, wrappedImage, H, roi.size());
	}

	// construct mask for copying the projection of the marker image to the original
	cv::Mat mask(imageColor.size(), CV_8U, cv::Scalar(0, 0, 0));
	cv::Mat whiteMask(vanImage.size(), CV_8U, cv::Scalar(255, 255, 255));
	cv::warpPerspective(whiteMask, whiteMask, H, wrappedImage.size());
	whiteMask.copyTo(mask(roi));
	wrappedImage.copyTo(imageColor(roi), mask(roi));
}
void Marker::findHomographyAndWriteImage(cv::Mat crop, cv::Mat marker, cv::Rect roi) const
{

	cv::resize(marker, marker, cv::Size(256, 256));
	cv::Mat cropResized;
	cv::resize(crop, cropResized, cv::Size(256, 256));

	cv::cvtColor(marker, marker, CV_RGB2GRAY);
	// use gaussian blur to get rid of noise
	cv::GaussianBlur(marker, marker, cv::Size(9, 9), 0, 0);
	cv::Mat temp1 = marker.clone();
	marker.empty();
	cv::bilateralFilter(temp1, marker, 5, 75, 75);

	// erode image to better detect points
	auto markerForLines = marker.clone();
	cv::threshold(markerForLines, markerForLines, 127, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// erode image for better shape detection; did the same for the original marker images but with
	// different values because of the different quality
	auto markerForShape = Moore::performErosion(marker, 0, 3);
	// threshold imagee for edge detection
	cv::threshold(markerForShape, markerForShape, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	auto bottomRightPointIndex = Marker::detectMarkerOrientation(markerForLines);
	auto markerNumber = -1;
	std::vector<cv::Vec4i> hierarchy;

	auto leo = cv::matchShapes(markerLeo, markerForShape, 2, 0.0);
	auto van = cv::matchShapes(markerVan, markerForShape, 2, 0.0);
	auto cropPoints = Helper::findCornersOnCrop(crop);

#if DEBUG_MODE
	std::cout << "leo match:" << leo << std::endl;
	std::cout << "van match: " << van << std::endl;
#endif
	if (cropPoints.size() > 0)
	{
		markerNumber = static_cast<int>(leo > van);
		if (markerNumber)
		{
			cv::imshow("marker vam", markerForLines);
		}
		else
		{
			cv::imshow("markerleo", markerForLines);
		}
		Marker::wrapMarkerOnImage(markerNumber, roi, cropPoints, bottomRightPointIndex);
	}


}


