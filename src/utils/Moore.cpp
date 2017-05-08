#include "Moore.hpp"
#include <opencv2/core/mat.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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


std::vector<cv::Point> Moore::computeBorders(cv::Mat image, int minLength, int maxLength)
{
	// construct white image
	cv::Mat path(cv::Mat(image.rows, image.cols, CV_THRESH_BINARY));
	std::vector<cv::Point> tempPoints;
	std::vector<cv::Point> points;
	path.setTo(0);
	bool inside = false;
	int i = 0;
	for (auto row = 0; row < image.rows; row++)
	{
		for (auto col = 0; col < image.cols; col++)
		{
			if (path.at<uchar>(row, col) == 255 &&  !inside)
			{
				inside = true;
			}
			else if (image.at<uchar>(row, col) == 255 && inside)
			{
				continue;
			}
			else if (image.at<uchar>(row, col) == 0 && inside)
			{
				inside = false;
			}
			else if (image.at<uchar>(row, col) == 255 && !inside)
			{
				cv::Point backtrack;
				cv::Point firstPixel = cv::Point(col, row);
				cv::Point error = cv::Point(-1, -1);
				cv::Point currentPixel = firstPixel;
				int iteration = 0; // first put a simple stopping criterion for test
				cv::Mat tempImage(cv::Mat(image.rows, image.cols, CV_THRESH_BINARY));
				tempImage.setTo(0);
				do
				{
					// we have higher access speed for seeing if a pixel is marked at a certain location so we keep the tempimage
					tempImage.at<uchar>(currentPixel) = 255; 
					tempPoints.push_back(currentPixel);
					currentPixel = findNextPixel(image, currentPixel, backtrack);
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
				if (iteration > minLength && iteration < maxLength)
				{
					// if the path meets the criteria print it's length and draw it on the image
					std::cout << "Boundary length: " << iteration << std::endl;
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

/* Helper function for performing erosion and dilation on a given matrix */
cv::Mat Moore::performErosion(cv::Mat img, int erosionElem, int erosionSize)
{
	int dilation_type;
	cv::Mat erosionResult;
	int erosion_type = 0;
	if (erosionElem == 0) { erosion_type = cv::MORPH_RECT; }
	else if (erosionElem == 1) { erosion_type = cv::MORPH_CROSS; }
	else if (erosionElem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

	cv::Mat element = cv::getStructuringElement(erosion_type,
		cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1),
		cv::Point(erosionSize, erosionSize));
	// Apply the dilation operation
	cv::erode(img, erosionResult, element);
	return erosionResult;
}

cv::Mat Moore::performDilation(cv::Mat img, int dilationElem, int dilationSize)
{
	int dilation_type = 0;
	cv::Mat dilationResult;
	if (dilationElem == 0) { dilation_type = cv::MORPH_RECT; }
	else if (dilationElem == 1) { dilation_type = cv::MORPH_CROSS; }
	else if (dilationElem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

	cv::Mat element = getStructuringElement(dilation_type,
		cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1),
		cv::Point(dilationSize, dilationSize));
	// Apply the dilation operation
	dilate(img, dilationResult, element);
	return dilationResult;
}