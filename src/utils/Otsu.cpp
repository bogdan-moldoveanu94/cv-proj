#include "Otsu.hpp"
#include <opencv2/core/mat.hpp>

int Otsu::computeTreshold(std::vector<int> histogram, int maxIntensity, int numberOfPixels)
{
	float  var_max, sum, sum_B, q1, q2, miu1, miu2, sigma_t;
	var_max = sum = sum_B = q1 = q2 = miu1 = miu2 = sigma_t = 0.0;
	int threshold = 0;
	for (int i = 0; i <= maxIntensity; i++) {
		sum += i * histogram.at(i);
	}

	for (auto t = 0; t <= maxIntensity; t++) {
		q1 += histogram.at(t);
		if (q1 == 0) {
			continue;
		}

		q2 = numberOfPixels - q1;
		if (q2 == 0)
			break;

		// update miu_i(t)

		sum_B += t * histogram.at(t);
		miu1 = sum_B / q1;
		miu2 = (sum - sum_B) / q2;

		sigma_t = q1 * q2 * ((miu1 - miu2) * (miu1 - miu2));
		if (sigma_t > var_max) {
			threshold = t;
			var_max = sigma_t;
		}
	}
	return threshold;
}

cv::Mat Otsu::computeTresholdOnImage(cv::Mat image, std::vector<int> histogram, int maxIntensity, int numberOfPixels)
{
	auto threshold = computeTreshold(histogram, maxIntensity, numberOfPixels);
	return image > threshold;
}

Otsu::~Otsu()
{
	
}
