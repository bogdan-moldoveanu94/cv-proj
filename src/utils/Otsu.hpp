#pragma once
#include <vector>

namespace cv {
	class Mat;
}

class Otsu
{
public:
	static int computeTreshold(std::vector<int> histogram, int maxIntensity, int numberOfPixels);
	static cv::Mat computeTresholdOnImage(cv::Mat image, std::vector<int> histogram, int maxIntensity, int numberOfPixels);
};

