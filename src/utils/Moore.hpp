#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>

namespace cv {
	class Mat;
}

class Moore
{
	std::vector<cv::Point> getOrderedNeighbours(cv::Point firstNeighbour);

	cv::Point findNextPixel(cv::Mat image, cv::Point currentPixel, cv::Point& backtrack);
	
public:
	cv::Mat computeBorders(cv::Mat image_padded);
};
