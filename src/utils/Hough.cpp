#include "Hough.hpp"
#include <opencv2/core/mat.hpp>

static int _width;
static int _height;
static int bins = 180; // chose no. of bins to be 180, one for each degree
static double houghH;
static double accuH = houghH * 2.0;
static const int accuW = 180;


double deg2rad(double degrees) {
	return degrees * 4.0 * atan(1.0) / 180.0;
}


std::vector<int> Hough::transform(cv::Mat image)
{
	
	_width = image.cols;
	_height = image.rows;
	houghH = ((sqrt(2.0) * double(_height > _width ? _height : _width)) / 2.0);
	accuH = houghH * 2.0;
	// create accumulator

	std::vector<int> accu(accuH * accuW + accuW); 
	double center_x = _width / 2;
	double center_y = _height / 2;
					
	// loop over all pixels in the image and increment accumulator at (r, theta)
	for (auto row = 0; row < _height; row++) {
		for (auto col = 0; col < _width; col++) 
			// see if pixel is dark enough
			if(image.at<uchar>(row, col) > 250)
			{
				for (auto theta = 0; theta < bins; theta++)
				{
					// radDist = x.cos(theta) + y.sin(theta)
					// compute from the center of the image
					auto radialDistance = (double(col) - center_x) * cos(deg2rad(double(theta))) + (double(row) - center_y) * sin(deg2rad(double(theta)));
					// increment accumulator
					accu[int((round(radialDistance + houghH) * 180.0)) + theta]++;
				}
				
			}
		}
	return accu;
}


std::vector<std::pair<std::pair<cv::Point, cv::Point>, std::pair<int, int>>> Hough::getLines(cv::Mat image, int threshold)
{
	std::vector<std::pair<std::pair<cv::Point, cv::Point>, std::pair<int, int>>> lines;

	auto accumulator = Hough::transform(image);
	// create accumulator

	if(accumulator.empty())
	{
		// return empty vector
		return lines; 
	}
	// loop over accumulator
	for(auto radialDistance = 0; radialDistance<accuH; radialDistance++)
	{
		for(auto theta = 0; theta< accuW; theta++)
		{
			// check if accumulator value is over required threshold
			if(int(accumulator[(radialDistance * accuW) + theta]) >= threshold)
			{
				// check for local maxima in a 9x9 square around it
				auto max = accumulator[(radialDistance*accuW) + theta];
				for(auto ly=-4; ly<=4;ly++)
				{
					for(auto lx = -4; lx<=4; lx++)
					{
						if((ly + radialDistance >= 0 && ly + radialDistance<accuH) && (lx + theta >= 0 && lx + theta<accuW))
						{
							if(int(accumulator[((radialDistance + ly) * accuW) + (theta + lx)]) > max)
							{
								max = accumulator[((radialDistance + ly)*accuW) + (theta + lx)];
								ly = lx = 5;
							}
						}
					}
				}
				// if already detected max is bigger just continue
				if(max >int(accumulator[(radialDistance * accuW) + theta]))
				{
					continue;
				}
				// if the acc value is above the threshold then compute the points coordinates and add them to the lines vector
				int x1, y1, x2, y2;
				x1 = y1 = x2 = y2 = 0;
				if(theta >= 45 && theta <= 135)
				{

					x1 = 0;
					y1 = (double(radialDistance - (accuH / 2)) - ((x1 - (_width / 2)) * cos(deg2rad(theta)))) / sin(deg2rad(theta)) + (_height / 2);
					x2 = _width;
					y2 = (double(radialDistance - (accuH / 2)) - ((x2 - (_width / 2)) * cos(deg2rad(theta)))) / sin(deg2rad(theta)) + (_height / 2);
				}
				else
				{
					y1 = 0;
					x1 = (double(radialDistance - (accuH / 2)) - ((y1 - (_height / 2)) * sin(deg2rad(theta)))) / cos(deg2rad(theta)) + (_width / 2);
					y2 = _height;
					x2 = (static_cast<double>(radialDistance - (accuH / 2)) - ((y2 - (_height / 2)) * sin(deg2rad(theta)))) / cos(deg2rad(theta)) + (_width / 2);
				}
				// make a std::pair w/ the coordinate of the points
				auto pointPair = std::make_pair(cv::Point(x1, y1), cv::Point(x2, y2));
				// we are interested only in the sign of rho for empirically computing the speed of the car in assignmnet 3 so we return just that
				auto r = radialDistance > accuH / 2 ? -1 : 1; 
				// make a pair of polar coords w/ the mention that r is just the sign
				auto polarCoords = std::make_pair(r, theta);
				lines.push_back(std::pair<std::pair<cv::Point, cv::Point>,std::pair< int, int>>(pointPair, polarCoords));
			}
		}
	}
	return lines;
}


