#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
#include <unordered_map>

namespace cv {
	class Mat;
}

class Moore
{
private:
	static std::vector<cv::Point> getOrderedNeighbours(cv::Point firstNeighbour);
	static cv::Point findNextPixel(cv::Mat image, cv::Point currentPixel, cv::Point& backtrack);
	static cv::Mat padImage(cv::Mat image);
	
public:
	std::vector<cv::Point> computeBorders(cv::Mat image_padded) const;
};
