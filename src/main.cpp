#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "utils/Histogram.hpp"
#include "utils/Otsu.hpp"
#include <tuple>

using namespace cv;
using namespace std;

Mat image_rgb, image_grayscale, image_padded;
const Scalar BLACK = cv::Scalar(0, 0, 0);
const Scalar WHITE = cv::Scalar(255, 255, 255);
const int MIN_COMPONENT_LENGTH = 100;
const int MAX_COMPONENT_LENGTH = 200;

std::tuple<int, int> findStartPixel(Mat image, std::tuple<int, int>& backtrack)
{
	for (auto row = 0; row < image.rows; row++)
	{
		for (auto col = 0; col <image.cols; col++)
		{
			backtrack = std::make_tuple(row, col);
			auto test = image.at<uchar>(row, col);
			if (image.at<uchar>(row, col) == 255)
			{
				auto val = std::make_tuple(row, col);
				return val;
			}
		}
	}
	return {};
}

std::vector<tuple<int, int>> getOrderedNeighbours(std::tuple<int, int> firstNeighbour)
{
	std::vector<std::tuple<int, int>> neighbours = { std::make_tuple(-1,-1), std::make_tuple(-1, 0), std::make_tuple(-1,1),
		std::make_tuple(0, 1), std::make_tuple(1, 1), std::make_tuple(1, 0), std::make_tuple(1,-1), std::make_tuple(0,-1) };
	std::vector<std::tuple<int, int>> orderedNeighbours;
	auto startingNeighbour = 0;
	for(auto i = 0; i < neighbours.size(); i++)
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

template<typename T>
std::tuple<T, T> operator + (const std::tuple<T, T> & lhs, const std::tuple<T, T> & rhs)
{
	return std::tuple<T, T>(std::get<0>(lhs) + std::get<0>(rhs), std::get<1>(lhs) + std::get<1>(rhs));
}

template<typename T>
std::tuple<T, T> operator - (const std::tuple<T, T> & lhs, const std::tuple<T, T> & rhs)
{
	return std::tuple<T, T>(std::get<0>(lhs) - std::get<0>(rhs), std::get<1>(lhs) - std::get<1>(rhs));
}

std::tuple<int, int> findNextPixel(cv::Mat image, std::tuple<int, int> currentPixel, std::tuple<int, int>& backtrack)
{
	auto startingOffset = backtrack;
	auto orderedNeighbours = getOrderedNeighbours(startingOffset);

	for (auto i = 0; i < orderedNeighbours.size(); ++i)
	{
		auto pos = currentPixel + orderedNeighbours[i];
		if (image.at<uchar>(std::get<0>(pos), std::get<1>(pos)))
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
	return make_tuple(-1, -1);
	std::cerr << "No next pixel - this means there was a pixel that is not connected to anything!" << std::endl;
	exit(-1);


}


cv::Mat Moore(Mat image_padded)
{
	// construct white image
	Mat path(Mat(image_padded.rows, image_padded.cols, CV_THRESH_BINARY));
	path.setTo(0);
	cv::imwrite("begin.png", path);

	bool inside = false;
	int i = 0;
	for(auto row = 0; row < image_padded.rows; row++)
	{
		for(auto col = 0; col < image_padded.cols; col++)
		{
			if(path.at<uchar>(row, col) == 255 && !inside)
			{
				inside = true;
			} 
			else if(image_padded.at<uchar>(row, col) == 255 && inside)
			{
				continue;
			}
			else if(image_padded.at<uchar>(row, col) == 0 && inside)
			{
				inside = false;
			}
			else if(image_padded.at<uchar>(row, col) == 255 && !inside)
			{
				std::tuple<int, int> backtrack;
				std::tuple<int, int> previousBacktrack; // for jacobi stopping criterion
				std::tuple<int, int> firstPixel = std::make_tuple(row, col);
				std::tuple<int, int> error = std::make_tuple(-1, -1);
				std::tuple<int, int> currentPixel = firstPixel;
				int iteration = 0; // first put a simple stopping criterion for test

				do
				{
					previousBacktrack = backtrack;

					path.at<uchar>(std::get<0>(currentPixel), std::get<1>(currentPixel)) = 255; // mark pixel as black on image; will change to a vector of points
					currentPixel = findNextPixel(image_padded, currentPixel, backtrack);
					if(currentPixel == error)
					{
						inside = true;
						break;
					}
					//cout << "inside the loop: " << i << endl;
					//i++;
					if(currentPixel == firstPixel)
					{
						iteration++;
					}
					
				} while (currentPixel != firstPixel && backtrack != previousBacktrack && iteration < 4);

				path.at<uchar>(std::get<0>(firstPixel), std::get<1>(firstPixel)) = 0;
				inside = true;
			}
		}
	}
	//std::tuple<int, int> backtrack;
	//std::tuple<int, int> previousBacktrack; // for jacobi stopping criterion
	//std::tuple<int, int> firstPixel = findStartPixel(image_padded, backtrack);

	//std::tuple<int, int> currentPixel = firstPixel;
	//do
	//{
	//	previousBacktrack = backtrack;
	//	path.at<uchar>(std::get<0>(currentPixel), std::get<1>(currentPixel)) = 255; // mark pixel as black on image; will change to a vector of points
	//	//cout << (int)path.at<uchar>((std::get<0>(currentPixel), std::get<1>(currentPixel))) << endl;
	//	currentPixel = findNextPixel(image_padded, currentPixel, backtrack);
	//	//cv::imshow("123", path);
	//	//std::cout << "Current pixel: " << currentPixel << " with backtrack: " << backtrack << std::endl;
	//	if(std::get<0>(currentPixel) == std::get<0>(firstPixel) && std::get<1>(currentPixel) == std::get<1>(firstPixel))
	//	{
	//		cout << " HERE0";
	//	}
	//	std::cout << "crt(" << std::get<0>(currentPixel) << ", " << std::get<1>(currentPixel) << ") vs (" << std::get<0>(firstPixel) << ", " << std::get<1>(firstPixel) << ")" << endl;
	//} while (currentPixel != firstPixel && backtrack != previousBacktrack);

	//// Close the loop
	//path.at<uchar>(std::get<0>(firstPixel), std::get<1>(firstPixel)) = 0;

	return path;

}


int main(int argc, char* argv[])
{
	image_rgb = imread(argv[1], 1);
	if (image_rgb.empty()) {
		cout << "Cannot read image ";
		return -1;
	}
	// convert image to grayscale
	cvtColor(image_rgb, image_grayscale, CV_RGB2GRAY);

#pragma region compute histogram

	Histogram histogramObj;
	vector<int> histogram = histogramObj.computeHistogramVector(image_grayscale);

	auto histImage = histogramObj.computeHistogramImage(image_grayscale, 256, 256, 256);
#pragma endregion


#pragma region otsu threshold computation
	auto numberOfPixels = image_grayscale.total();
	auto max_intensity = 255;

	Otsu otsuObj;

	// just to check that the threashold has the right value
	//auto otsuTreshold = otsuObj.computeTreshold(histogram, max_intensity, numberOfPixels);
	//std::cout << "value from my impl: " << otsuTreshold << endl;

	auto output = otsuObj.computeTresholdOnImage(image_grayscale, histogram, max_intensity, numberOfPixels);
#pragma endregion

	// use built-in thresholding function from opencv in order to test our implementation

	Mat otsu;
	double th_val = cv::threshold(image_grayscale, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//std::cout << "value from built in: " << th_val << endl;
	otsu = otsu > th_val;
	cv::imwrite("otsucv.png", otsu);


	// pad image w/ 1 pixel white border
	auto border = cv::Scalar(0);
	std::cout << "otsu image: " << "rows: " << output.rows << " cols: " << output.cols << endl;
	cv::copyMakeBorder(otsu, image_padded, 1, 1, 1, 1, BORDER_CONSTANT, border);
	std::cout << "padded image: " << "rows: " << image_padded.rows << " cols: " << image_padded.cols;



	// construct white image
	Mat image_border(Mat(image_padded.rows, image_padded.cols, CV_8U));
	image_border.setTo(255);

	//cv::imwrite("hist.png", histImage);
	cv::imwrite("output.png", output);

	// comment out debugging code
	//cv::namedWindow("image", 1);
	auto output1 = Moore(image_padded);
		cv::imshow("image", output1);
	cv::imwrite("border.png", output1);

	//cv::imshow("image2", otsu);
	//cv::imshow("hist", histImage);
	cv::waitKey();

	return 0;

}





