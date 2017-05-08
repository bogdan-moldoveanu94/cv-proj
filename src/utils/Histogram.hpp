#pragma once
#include <vector>

namespace cv {
	class Mat;
}

class Histogram
{
public:
	static std::vector<int> computeHistogramVector(cv::Mat image);
	static cv::Mat computeHistogramImage(cv::Mat inputImage, int histSize, int histWidth, int histHeight);
	~Histogram();
};
