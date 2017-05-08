#pragma once
#include <opencv2/core/mat.hpp>

namespace cv {
	class Mat;
}

class Hough
{
	
public:
	static std::vector<int> transform(cv::Mat image);
	static std::vector<std::pair<std::pair<cv::Point, cv::Point>, std::pair<int, int>>> getLines(cv::Mat image, int treshold);
};
