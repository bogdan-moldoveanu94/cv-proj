#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

class Marker
{
public:
	Marker();
	static void preProcessMarkers();
	~Marker();
	cv::Mat preProcessImage(cv::Mat image);
	std::vector<std::vector<cv::Point>> findCandidateContours(cv::Mat image) const;
	cv::Rect convertContourToRoi(std::vector<cv::Point>) const;
	std::vector<cv::Point2f> orderContourPoints(std::vector<cv::Point> contours) const;
	static bool detectStrongLinePoints(cv::Mat image, std::vector<std::vector<cv::Point2f>>* points);
	void findHomographyFeatures(cv::Mat crop, cv::Mat marker, cv::Mat originalImage, cv::Rect roi, cv::VideoWriter outputVideo, double fps);
	static std::vector<cv::Point2f> markerCornerPoints;
private:
	static cv::Mat markerLeo, markerVan, vanImage, monaImage, imageColor;
	cv::Mat imageGrayscale;

};

