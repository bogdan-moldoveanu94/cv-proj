#pragma once
#include <vector>

namespace cv {
	class Mat;
}

std::tuple<int, int> findStartPixel(cv::Mat image, std::tuple<int, int>& backtrack);

std::vector<std::tuple<int, int>> getOrderedNeighbours(std::tuple<int, int> firstNeighbour);

template<typename T>
std::tuple<T, T> operator + (const std::tuple<T, T> & lhs, const std::tuple<T, T> & rhs);

template<typename T>
std::tuple<T, T> operator - (const std::tuple<T, T> & lhs, const std::tuple<T, T> & rhs);

std::tuple<int, int> findNextPixel(cv::Mat image, std::tuple<int, int> currentPixel, std::tuple<int, int>& backtrack);

cv::Mat Moore(cv::Mat image_padded, cv::Mat image_color);