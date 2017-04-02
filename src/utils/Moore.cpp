#include "Moore.hpp"
#include <opencv2/core/mat.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui.hpp>

template<typename T>
cv::Point operator -(const cv::Point & lhs, const cv::Point & rhs)
{
	return cv::Point(lhs.x + rhs.x, lhs.y + rhs.y);
}

template<typename T>
cv::Point operator + (const cv::Point & lhs, const cv::Point & rhs)
{
	return cv::Point(lhs.x - rhs.x, lhs.y - rhs.y);
}


cv::Point findStartPixel(cv::Mat image, cv::Point& backtrack)
{
	for (auto row = 0; row < image.rows; row++)
	{
		for (auto col = 0; col <image.cols; col++)
		{
			backtrack = cv::Point(row, col);
			auto test = image.at<uchar>(row, col);
			if (image.at<uchar>(row, col) == 255)
			{
				auto val = cv::Point(row, col);
				return val;
			}
		}
	}
	return{};
}

std::vector<cv::Point> getOrderedNeighbours(cv::Point firstNeighbour)
{
	std::vector<cv::Point> neighbours = { cv::Point(-1,-1),  cv::Point(-1, 0),  cv::Point(-1,1),
		cv::Point(0, 1),  cv::Point(1, 1),  cv::Point(1, 0),  cv::Point(1,-1),  cv::Point(0,-1) };
	std::vector<cv::Point> orderedNeighbours;
	auto startingNeighbour = 0;
	for (auto i = 0; i < neighbours.size(); i++)
	{
		if (neighbours[i] == firstNeighbour)
		{
			startingNeighbour = i;
			break;
		}
	}

	// Add the reamining offsets to the end of the list
	for (auto i = startingNeighbour + 1; i < neighbours.size(); ++i)
	{
		orderedNeighbours.push_back(neighbours[i]);
	}

	// Add the reamining offsets from the beginning of the list
	for (auto i = 0; i < startingNeighbour; ++i)
	{
		orderedNeighbours.push_back(neighbours[i]);
	}

	return orderedNeighbours;
}

cv::Point findNextPixel(cv::Mat image, cv::Point currentPixel, cv::Point& backtrack)
{
	auto startingOffset = backtrack;
	auto orderedNeighbours = getOrderedNeighbours(startingOffset);

	for (auto i = 0; i < orderedNeighbours.size(); ++i)
	{
		//std::cout << currentPixel.x << " " << currentPixel.y << std::endl;
		//std::cout << orderedNeighbours[i].x << " " << orderedNeighbours[i].y << std::endl;
		auto pos = currentPixel + orderedNeighbours[i];
		//std::cout << pos.x << " " << pos.y << std::endl;
		//std::cout << std::endl;
		if (image.at<uchar>(pos) == 255)
		{
			if (i != 0)
			{
				backtrack = (currentPixel + orderedNeighbours[i - 1]) - (currentPixel + orderedNeighbours[i]);
			}
			else
			{
				backtrack = (currentPixel + startingOffset) - (currentPixel + orderedNeighbours[i]);
			}

			return currentPixel + orderedNeighbours[i];
		}


	}
	return cv::Point(-1, -1);
}


cv::Mat Moore(cv::Mat image_padded, cv::Mat image_color)
{
	// construct white image
	cv::Mat path(cv::Mat(image_padded.rows, image_padded.cols, CV_THRESH_BINARY));
	path.setTo(0);
	cv::imwrite("begin.png", path);
	auto pct1 = cv::Point(1, 11);
	auto pct2 = cv::Point(1, 10);
	//std::cout << pct1 + pct2<< std::endl;
	bool inside = false;
	int i = 0;
	for (auto row = 0; row < image_padded.rows; row++)
	{
		for (auto col = 0; col < image_padded.cols; col++)
		{
			if (path.at<uchar>(row, col) == 255 && !inside)
			{
				inside = true;
			}
			else if (image_padded.at<uchar>(row, col) == 255 && inside)
			{
				continue;
			}
			else if (image_padded.at<uchar>(row, col) == 0 && inside)
			{
				inside = false;
			}
			else if (image_padded.at<uchar>(row, col) == 255 && !inside)
			{
				cv::Point backtrack;
				cv::Point previousBacktrack;
				//std::tuple<int, int> previousBacktrack; // for jacobi stopping criterion
				//cv::Point firstPixel = cv::Point(row, col);
				// switch rows and cols for cv::point fmm
				cv::Point firstPixel = cv::Point(col, row);
				std::cout << " first pixel is: " << firstPixel.x << " " << firstPixel.y << std::endl;
				cv::Point error = cv::Point(-1, -1);
				cv::Point currentPixel = firstPixel;
				int iteration = 0; // first put a simple stopping criterion for test
				cv::Mat tempImage(cv::Mat(image_padded.rows, image_padded.cols, CV_THRESH_BINARY));
				tempImage.setTo(0);
				do
				{
					tempImage.at<uchar>(currentPixel) = 255; // mark pixel as white on image; will change to a vector of points
					//image_color.at<cv::Vec3b>(currentPixel) = cv::Vec3b(0, 0, 0);
					currentPixel = findNextPixel(image_padded, currentPixel, backtrack);
					if (currentPixel == error)
					{
						inside = true;
						break;
					}
					iteration++;
				} while (currentPixel != firstPixel);
				std::cout << iteration << std::endl;
				tempImage.at<uchar>(firstPixel) = 0;
				//path.at<uchar>(firstPixel) = 255;
				std::cout << "outside the loop";
				if (iteration > 100 && iteration < 200)
				{
					path += tempImage;
				}
				inside = true;
			}
		}
	}
	std::cout << "done";
	cv::imshow("wat", path);
	cv::waitKey();
	return path;


}
