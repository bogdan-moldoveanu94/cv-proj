#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#define DEBUG_MODE 0
#define MARKER_WIDTH 256
#define MARKER_HEIGHT 256

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
	int computerOrientationFromLinePoints(cv::Mat image, std::vector<std::vector<cv::Point2f>> points) const;
	static bool detectStrongLinePoints(cv::Mat image, std::vector<std::vector<cv::Point2f>>* points);
	int detectMarkerOrientation(cv::Mat image) const;
	void wrapMarkerOnImage(int markerNumber, cv::Rect roi, std::vector<cv::Point2f> cropPoints, int bottomRightPointIndex) const;
	void findHomographyAndWriteImage(cv::Mat crop, cv::Mat marker, cv::Rect roi) const;
	static std::vector<cv::Point2f> markerCornerPoints;
	static cv::Mat imageColor;
	static cv::Rect canonicalRoi;

private:
	static cv::Mat markerLeo, markerVan, vanImage, monaImage;
	cv::Mat imageGrayscale;
	bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f& r) const;
};

