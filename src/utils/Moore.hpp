#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>

namespace cv {
	class Mat;
}

class Moore
{
private:
	std::vector<cv::Point> getOrderedNeighbours(cv::Point firstNeighbour);
	cv::Point findNextPixel(cv::Mat image, cv::Point currentPixel, cv::Point& backtrack);
	cv::Mat padImage(cv::Mat image);
	
public:
	std::vector<cv::Point> computeBorders(cv::Mat image_padded);
};
