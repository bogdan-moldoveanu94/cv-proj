#include "Hough.hpp"
#include <opencv2/core/mat.hpp>
#include <iostream>

double deg2rad(double degrees) {
	return degrees * 4.0 * atan(1.0) / 180.0;
}


unsigned* Hough::transform(cv::Mat image)
{
	
	auto width = image.cols;
	auto height = image.rows;
	// create accumulator
	double houghH = ((sqrt(2.0) * double(height > width ? height : width)) / 2.0);
	auto accuH = houghH * 2.0;
	auto accuW = 180;

	auto accu = (unsigned int*)calloc(accuH * accuW, sizeof(unsigned int));
	double center_x = width / 2;
	double center_y = height / 2;
					

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) 
			if(image.at<uchar>(row, col) > 250)
			{
				for (auto t = 0; t< 180; t++)
				{
					auto r = (double(col) - center_x) * cos(deg2rad(double(t))) + (double(row) - center_y) * sin(deg2rad(double(t)));
					accu[int((round(r + houghH) * 180.0)) + t]++;
				}
				
			}
		}
	return accu;
}

std::vector<std::pair<cv::Point, cv::Point>> Hough::getLines(cv::Mat image, int threshold)
{
	std::vector<std::pair<cv::Point, cv::Point>> lines;

	// to remove this and get them from the above function

	auto width = image.cols;
	auto height = image.rows;
	// create accumulator
	double houghH = ((sqrt(2.0) * double(height > width ? height : width)) / 2.0);
	auto accuH = houghH * 2.0;
	auto accuW = 180;

	auto accumulator = Hough::transform(image);
	if(accumulator == nullptr)
	{
		return lines;
	}
	for(auto r = 0; r<accuH; r++)
	{
		for(auto t = 0; t< accuW; t++)
		{
			if((int)accumulator[(r*accuW) + t] >= threshold)
			{
				// check for local maxima
				auto max = accumulator[(r*accuW) + t];
				for(auto ly=-4; ly<=4;ly++)
				{
					for(auto lx = -4; lx<=4; lx++)
					{
						if((ly + r >= 0 && ly + r<accuH) && (lx + t >= 0 && lx + t<accuW))
						{
							if((int)accumulator[((r + ly)*accuW) + (t + lx)] > max)
							{
								max = accumulator[((r + ly)*accuW) + (t + lx)];
								ly = lx = 5;
							}
						}
					}
				}
				if(max >(int)accumulator[(r*accuW) + t])
				{
					continue;
				}
				int x1, y1, x2, y2;
				x1 = y1 = x2 = y2 = 0;

				if(t >= 45 && t <= 135)
				{

					x1 = 0;
					y1 = ((double)(r - (accuH / 2)) - ((x1 - (width / 2)) * cos(deg2rad(t)))) / sin(deg2rad(t)) + (height / 2);
					x2 = width - 0;
					y2 = ((double)(r - (accuH / 2)) - ((x2 - (width / 2)) * cos(deg2rad(t)))) / sin(deg2rad(t)) + (height / 2);
				}
				else
				{
					y1 = 0;
					x1 = ((double)(r - (accuH / 2)) - ((y1 - (height / 2)) * sin(deg2rad(t)))) / cos(deg2rad(t)) + (width / 2);
					y2 = image.rows - 0;
					x2 = ((double)(r - (accuH / 2)) - ((y2 - (height / 2)) * sin(deg2rad(t)))) / cos(deg2rad(t)) + (width / 2);
				}
				lines.push_back(std::pair< cv::Point, cv::Point >(cv::Point(x1, y1), cv::Point(x2, y2)));
			}
		}
	}
	std::cout << "lines: " << lines.size() << " " << threshold << std::endl;
	return lines;

}


