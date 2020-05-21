// Standard C++
#include <stdio.h>
#include <iostream>

// OpenCV Imports
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // OpenCV Core Functionality
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

	/****** TASK 1 ******/
	// Read videos
	VideoCapture capbg("belt_bg.wmv");
	if (!capbg.isOpened())
	{
		//error in opening video of background
		cerr << "Unable to open: " << "belt_bg.wmv" << endl;
		return 0;
	}
	
	float numframesbg = 359; 
	double fps = 40;
	double wait = floor(1000 / fps);

	VideoCapture capfg("belt_fg.wmv");
	if (!capfg.isOpened())
	{
		//error in opening video of foreground
		cerr << "Unable to open: " << "belt_fg.wmv" << endl;
		return 0;
	}

	// Show videos
	while (1)
	{
		Mat frame;
		capbg >> frame;

		if (frame.empty())
			break;

		imshow("Frame", frame);

		char c = (char)waitKey(wait); //display at fps
		if (c == 27)
			break;
	}
	destroyAllWindows();

	while (1)
	{
		Mat frame;
		capfg >> frame;

		if (frame.empty())
			break;

		imshow("Frame", frame);

		char c = (char)waitKey(wait);
		if (c == 27)
			break;
	}
	destroyAllWindows();


	/******* TASK 2 ***********/
	// Create new objects to reset frame count
	capbg.~VideoCapture();
	VideoCapture cap2bg("belt_bg.wmv");
	if (!cap2bg.isOpened())
	{
		//error in opening video of background
		cerr << "Unable to open: " << "belt_bg.wmv" << endl;
		return 0;
	}

	capfg.~VideoCapture();
	VideoCapture cap2fg("belt_fg.wmv");
	if (!cap2fg.isOpened())
	{
		//error in opening video of foreground
		cerr << "Unable to open: " << "belt_fg.wmv" << endl;
		return 0;
	}

	int i = 3; // size of averaging filter

	Mat frame2;
	cap2bg >> frame2; //storing for sizing purposes later
	if (frame2.empty())
		return 0;

	cap2bg.~VideoCapture();
	VideoCapture cap3bg("belt_bg.wmv");
	if (!cap3bg.isOpened())
	{
		//error in opening video of background
		cerr << "Unable to open: " << "belt_bg.wmv" << endl;
		return 0;
	}

	// Find mean
	Mat mean = Mat::zeros(frame2.rows, frame2.cols, CV_32FC1);
	Mat stdv = Mat::zeros(frame2.rows, frame2.cols, CV_32FC1);
	Mat diff = Mat::zeros(frame2.rows, frame2.cols, CV_32FC1);
	Mat contours = Mat::zeros(frame2.rows, frame2.cols, CV_32FC1);
	Mat hierarchy = Mat::zeros(frame2.rows, frame2.cols, CV_32FC1);

	while (1)
	{
		Mat frame;
		cap3bg >> frame;

		if (frame.empty())
			break;

		cvtColor(frame, frame, COLOR_RGB2GRAY);

		GaussianBlur(frame, frame, Size(i, i), 0, 0);

		frame.convertTo(frame, CV_32FC1);

		for (int r = 0; r < frame.rows; r++)
		{
			for (int c = 0; c < frame.cols; c++)
				mean.at<float>(r, c) += (frame.at<float>(r, c) / 359);
		}
	}

	cap3bg.~VideoCapture();
	VideoCapture cap4bg("belt_bg.wmv");
	if (!cap4bg.isOpened())
	{
		//error in opening video of background
		cerr << "Unable to open: " << "belt_bg.wmv" << endl;
		return 0;
	}

	// Find std
	while (1)
	{
		Mat frame;
		cap4bg >> frame;

		if (frame.empty())
			break;

		cvtColor(frame, frame, COLOR_RGB2GRAY);

		GaussianBlur(frame, frame, Size(i, i), 0, 0);

		frame.convertTo(frame, CV_32FC1);

		diff = frame - mean;

		for (int r = 0; r < frame.rows; r++)
		{
			for (int c = 0; c < frame.cols; c++)
				stdv.at<float>(r, c) += (diff.at<float>(r, c)) * (diff.at<float>(r, c)) / 359;
		}
	}

	// Subtract background and draw contours
	float thresh = 3;
	int count = 0;
	int avgdefects = 0;
	int highweight = 10;
	int lowweight = 5;
	int extralowweight = 1;
	int cnut = 0;
	int cpeg = 0;
	int cpipe = 0;
	int cprong = 0;
	int cqueens = 0;
	int cwasher = 0;
	float area = 0;
	float currarea = 0;
	float extent = 0;

	while (1)
	{
		/***** background subtraction *******/
		Mat frame;
		cap2fg >> frame;

		if (frame.empty())
			break;

		cvtColor(frame, frame, COLOR_RGB2GRAY);

		GaussianBlur(frame, frame, Size(i, i), 0, 0);

		frame.convertTo(frame, CV_32FC1);

		for (int r = 0; r < frame.rows; r++)
		{
			for (int c = 0; c < frame.cols; c++)
			{
				if (abs(frame.at<float>(r, c) - mean.at<float>(r, c)) > (thresh * stdv.at<float>(r, c)))
					frame.at<float>(r, c) = (float)255;
				else
					frame.at<float>(r, c) = (float)0;
			}
		}

		frame.convertTo(frame, CV_8UC1);
		imshow("Background Subtracted", frame);
		/************************************/


		/***** Drawing contours *************/
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy, defects;
		vector<int> hull;

		findContours(frame, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		Mat drawing = Mat::zeros(frame.size(), CV_8UC1);
		area = 0;
		int j = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(255, 255, 255);
			drawContours(drawing, contours, i, color, 1, 8, hierarchy, 2, Point(0, 0));

			currarea = contourArea(contours[i]);
			if (currarea > area)
			{
				area = currarea;
				j = i;
			}

		}

		drawing.convertTo(drawing, CV_8UC1);
		namedWindow("Contours", WINDOW_AUTOSIZE);
		imshow("Contours", drawing);
		char c = (char)waitKey(1);
		if (c == 27)
			break;
		/***********************************/

		/***** Feature Extraction **********/
		// Convex hull defect extraction
		if (area > 2000)
		{
			convexHull(contours[j], hull);
			convexityDefects(contours[j], hull, defects);
			Rect rect = boundingRect(contours[j]);
			float rectarea = rect.width * rect.height;
			extent = area / rectarea;

			if (20 < defects.size() && defects.size() < 30)
				cnut += lowweight;
			else if (27 < defects.size() && defects.size() < 32)
				cpeg += lowweight;
			else if (10 < defects.size() && defects.size() < 15)
				cpipe += highweight;
			else if (15 < defects.size() && defects.size() < 22)
				cprong += lowweight;
			else if (15 < defects.size() && defects.size() < 20)
				cqueens += lowweight;
			else if (30 < defects.size() && defects.size() < 40)
				cwasher += highweight;

			// Contour area
			if (8900 < area && area < 11000)
				cnut += highweight;
			else if (3900 < area && area < 4300)
				cpeg += highweight;
			else if (2000 < area && area < 2400)
				cpipe += extralowweight;
			else if (7700 < area && area < 9200)
				cprong += highweight;
			else if (6400 < area && area < 7700)
				cqueens += highweight;
			else if (5000 < area && area < 5700)
				cwasher += highweight;

			// Extent
			if (0.71 < extent && extent < 0.74)
				cnut += extralowweight;
			else if (0.75 < extent && extent < 0.77)
				cpeg += lowweight;
			else if (0.37 < extent && extent < 0.45)
				cpipe += highweight;
			else if (0.44 < extent && extent < 0.47)
				cprong += highweight;
			else if (0.52 < extent && extent < 0.54)
				cqueens += highweight;
			else if (0.76 < extent && extent < 0.78)
				cwasher += lowweight;

			if (count > 10)
			{
				if (cnut > cpeg && cnut > cpipe && cnut > cprong && cnut > cqueens && cnut > cwasher)
					cout << "Class NUT" << endl;
				if (cpeg > cnut && cpeg > cpipe && cpeg > cprong && cpeg > cqueens && cpeg > cwasher)
					cout << "Class PEG" << endl;
				if (cpipe > cnut && cpipe > cpeg && cpipe > cprong && cpipe > cqueens && cpipe > cwasher)
					cout << "Class PIPE" << endl;
				if (cprong > cnut && cprong > cpeg && cprong > cpipe && cprong > cqueens && cprong > cwasher)
					cout << "Class PRONG" << endl;
				if (cqueens > cnut && cqueens > cpeg && cqueens > cpipe && cqueens > cprong && cqueens > cwasher)
					cout << "Class QUEENS" << endl;
				if (cwasher > cnut && cwasher > cpeg && cwasher > cpipe && cwasher > cprong && cwasher > cqueens)
					cout << "Class WASHER" << endl;
				count = 0;
				cnut = cpeg = cpipe = cprong = cqueens = cwasher = 0;
			}
			else
				count++;
		}
	}
	return 0;
}
