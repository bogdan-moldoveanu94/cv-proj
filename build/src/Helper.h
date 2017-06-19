#pragma once
#include <opencv2/core/base.hpp>
#include "../../src/utils/Hough.hpp"

class Helper
{
public:
	Helper();
	~Helper();
	static void findHomographyUsingContours(cv::Mat crop, cv::Mat marker);
	static void findHomographyMatrix(cv::Mat crop, cv::Mat marker, std::vector<cv::Point2f> convertedContours, std::vector<cv::Point2f> markerCornerPoints);
	static void findHomographyFeatures(cv::Mat crop, cv::Mat marker, std::vector<cv::Point2f> cropPoints, std::vector<cv::Point2f> markerPoints, cv::Mat originalImage, cv::Rect roi);
	static std::vector<cv::Point2f> findCornersOnCrop(cv::Mat crop);
};

