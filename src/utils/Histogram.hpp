#pragma once
#include <vector>

namespace cv {
	class Mat;
}

class Histogram
{
	std::vector<int>* histogram;
public:
	std::vector<int> computeHistogramVector(cv::Mat image);
	cv::Mat computeHistogramImage(cv::Mat inputImage, int histSize, int histWidth, int histHeight);
	~Histogram();
};
