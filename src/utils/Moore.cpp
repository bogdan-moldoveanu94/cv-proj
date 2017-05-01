#include "Moore.hpp"
#include <opencv2/core/mat.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui.hpp>

cv::Mat Moore::padImage(cv::Mat image)
{
	return image;
}

std::vector<cv::Point> Moore::getOrderedNeighbours(cv::Point firstNeighbour)
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

cv::Point Moore::findNextPixel(cv::Mat image, cv::Point currentPixel, cv::Point& backtrack)
{
	auto startingOffset = backtrack;
	auto orderedNeighbours = getOrderedNeighbours(startingOffset);

	for (auto i = 0; i < orderedNeighbours.size(); ++i)
	{
		auto pos = currentPixel + orderedNeighbours[i];
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


std::vector<cv::Point> Moore::computeBorders(cv::Mat image_padded)
{
	// construct white image
	cv::Mat path(cv::Mat(image_padded.rows, image_padded.cols, CV_THRESH_BINARY));
	std::vector<cv::Point> tempPoints;
	std::vector<cv::Point> points;
	path.setTo(0);
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
				cv::Point firstPixel = cv::Point(col, row);
				cv::Point error = cv::Point(-1, -1);
				cv::Point currentPixel = firstPixel;
				int iteration = 0; // first put a simple stopping criterion for test
				cv::Mat tempImage(cv::Mat(image_padded.rows, image_padded.cols, CV_THRESH_BINARY));
				tempImage.setTo(0);
				do
				{
					tempImage.at<uchar>(currentPixel) = 255; // mark pixel as white on image; will change to a vector of points
					tempPoints.push_back(currentPixel);
					currentPixel = findNextPixel(image_padded, currentPixel, backtrack);
					if (currentPixel == error)
					{
						inside = true;
						break;
					}
					//count the number of pixels from the path
					iteration++;
				} while (currentPixel != firstPixel);

				tempImage.at<uchar>(firstPixel) = 0; // close the loop
				tempPoints.push_back(firstPixel);
				if (iteration > 100 && iteration < 200)
				{
					// if the path meets the criteria print it's length and draw it on the image
					std::cout << iteration << std::endl;
					path += tempImage;
					points.insert(std::end(points), std::begin(tempPoints), std::end(tempPoints));
					tempPoints.clear();
				}
				else
				{
					tempPoints.clear();
				}
				inside = true;
			}
		}
	}
	return points;
}
