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
	static std::vector<cv::Point> computeBorders(cv::Mat image, int minLength, int maxLength);
	static cv::Mat performDilation(cv::Mat img, int dilationElem, int dilationSize);
	static cv::Mat performErosion(cv::Mat, int erosionElem, int erosionSize);
};
