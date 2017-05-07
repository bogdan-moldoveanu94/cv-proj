#pragma once
#include <opencv2/core/mat.hpp>

namespace cv {
	class Mat;
}

class Hough
{
	
public:
	static unsigned* transform(cv::Mat image);
	static std::vector<std::pair<cv::Point, cv::Point>> getLines(cv::Mat image, int treshold);
};
