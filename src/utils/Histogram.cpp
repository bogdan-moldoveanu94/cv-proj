#include "Histogram.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>


std::vector<int> Histogram::computeHistogramVector(cv::Mat image)
{
	std::vector<int> histogram(image.total());
	for (auto row = 0; row < image.rows; row++) {
		for (auto column = 0; column < image.cols; column++) {
			auto pos = 0xFF & image.at<uchar>(row, column);
			(histogram)[pos]++;
		}
	}
	return histogram;
}

cv::Mat Histogram::computeHistogramImage(cv::Mat inputImage, int histSize, int histWidth, int histHeight)
{
	auto hist = Histogram::computeHistogramVector(inputImage);
	auto bin_w = cvRound(double(histWidth) / histSize);
	cv::Mat outputImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	normalize(hist, hist, 0, outputImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (auto i = 1; i < histSize; i++) {
		line(outputImage, cv::Point(bin_w*(i - 1), histHeight - cvRound(hist.at(i - 1))),
		     cv::Point(bin_w*(i), histHeight - cvRound(hist.at(i))),
		     cv::Scalar(255, 0, 0), 2, 8, 0);
	}
	return outputImage;
}

Histogram::~Histogram()
{
}
