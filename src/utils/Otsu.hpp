#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>


class Otsu
{
public:
	static int computeTreshold(std::vector<int> histogram, int maxIntensity, int numberOfPixels);
	static cv::Mat computeTresholdOnImage(cv::Mat image, std::vector<int> histogram, int maxIntensity, int numberOfPixels);
	~Otsu();
};

