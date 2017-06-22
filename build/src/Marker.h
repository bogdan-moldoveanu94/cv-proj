#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

class Marker
{
public:
	Marker();
	~Marker();
	cv::Mat preProcessImage(cv::Mat image);
	std::vector<std::vector<cv::Point>> findCandidateContours(cv::Mat image) const;
	cv::Rect convertContourToRoi(std::vector<cv::Point>) const;
	std::vector<cv::Point2f> orderContourPoints(std::vector<cv::Point> contours) const;
	void findHomographyFeatures(cv::Mat crop, cv::Mat marker, std::vector<cv::Point2f> cropPoints, cv::Mat originalImage, cv::Rect roi, cv::VideoWriter outputVideo, std::vector<std::vector<cv::Point>> leoContour, std::vector<std::vector<cv::Point>> vanContour, cv::Mat leoMarker, cv::Mat vanMarker, cv::Mat canonicalMarkerOriginal);
private:
	static cv::Mat markerLeo, markerVan, vanImage, monaImage, imageColor;
	cv::Mat imageGrayscale;
	static std::vector<cv::Point2f> markerCornerPoints;
};

