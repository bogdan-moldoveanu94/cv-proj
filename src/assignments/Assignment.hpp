#pragma once
#pragma once
#include <vector>
#include <opencv2/videoio.hpp>

namespace cv {
	class Mat;
}

class Assignment
{
private:
	static double angleToVelocity(double angle, double rho, bool isRightSide);
public:
	static void runFirstAssignment(cv::Mat);
	static void runSecondAssignment(cv::Mat);
	static void runThirdAssignment(cv::VideoCapture capture);
};
