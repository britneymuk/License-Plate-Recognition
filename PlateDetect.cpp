#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include "Supp.h"

using namespace cv;
using namespace std;
using namespace tesseract;

Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

void charSegment(Mat src, Mat dst) { /// Character segmentation
	Mat border = Mat::zeros(src.size(), src.type()), content = Mat::zeros(src.size(), src.type()), intersection;
	// Create a border template
	border.row(0) = 255;
	border.row(border.rows - 1) = 255;
	border.col(0) = 255;
	border.col(border.cols - 1) = 255;
	// Define the minimum character height ratio 
	int minCharHeight;
	if (((float)src.rows / src.cols)<0.3) { // For Long plate
		minCharHeight = src.rows * 0.25;
	}
	else {// For Square plate
		minCharHeight = src.rows * 0.15;
	}

	// Remove license plate frame & noises
	vector<vector<Point> > contours;
	findContours(src, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<Rect> boundRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		Mat drawing = Mat::zeros(src.size(), src.type());
		drawContours(drawing, contours, (int)i, Scalar(255, 255, 255), FILLED);
		intersection = drawing & border;// If contour touched the border, remove it (ignore).
		if (!countNonZero(intersection) > 0) {// The one that didn't intersect with border
			if (!(boundRect[i].height < minCharHeight)) {// The one that fullfil the desired height
				drawContours(content, contours, (int)i, Scalar(255, 255, 255), FILLED);
			}
		}
	}
	
	dst = content & src;
}

void charsSetSegment(Mat src, vector<Mat>& dst, vector<Rect>& roi) {/// Characters set segmentation (Split plates)
	Mat border = Mat::zeros(src.size(), src.type()), content = Mat::zeros(src.size(), src.type()), intersection, temp;
	// Create a border template
	border.row(0) = 255;
	border.row(border.rows - 1) = 255;
	border.col(0) = 255;
	border.col(border.cols - 1) = 255;
	// Define the minimum character area
	int minCharArea= (src.rows * src.cols) * 0.004;
	// Define the minimum characters set area
	int minCharsArea = (src.rows * src.cols) * 0.01;

	// Remove license plate frames & noises
	dilate(src, temp, kernel);
	vector<vector<Point> > contours;
	findContours(temp, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<Rect> boundRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		int area = boundRect[i].area();
		Mat drawing = Mat::zeros(src.size(), src.type());
		drawContours(drawing, contours, (int)i, Scalar(255, 255, 255), FILLED);
		intersection = drawing & border; 
		if (!countNonZero(intersection) > 0) {// The one that didn't intersect with border
			if (area> minCharArea) { // The one that fullfil the desired character area
				rectangle(content, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 255), FILLED);
			}	
		}
	}
	
	// Link the character to characters set
	vector<Vec4i>	lines;
	HoughLinesP(content, lines, 1, (CV_PI / 180)*90, 80, 5, 60);// Hough Line
	for (size_t i = 0; i < lines.size(); i++) {
		line(content, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255, 255, 255));
	}	
	morphologyEx(content, content, MORPH_CLOSE, kernel);
	findContours(content, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	content = Mat::zeros(src.size(), src.type());
	for ( int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		if (boundRect[i].area() > minCharsArea) { // The one that fullfil the desired characters set area
			dst.push_back(src(Rect(boundRect[i])));
			roi.push_back(Rect(boundRect[i]));
			rectangle(content, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 255), FILLED);
		}
	}
}

int main() {

	string		inputImagePath,imgName;
	Mat			imgGray, imgHSV, imgBlur, imgResult, mask, imgCrop, temp;
	int const	noOfImagePerCol = 1, noOfImagePerRow = 2;
	Mat			largeWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
	Mat			kernel2;
	int			maxWpixels;


	// OCR initialization
	char* outText;
	TessBaseAPI* api = new TessBaseAPI();
	if (api->Init("tessdata", "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}

	// Input file
	ifstream inFile;
	inFile.open("in/inputFileNames.txt");
	if (!inFile) {
		cout << "\n\nERROR: Fail to open in/inputFileNames.txt for reading.\n\n" << endl;
		inFile.close();
		return 0;
	}
	else {
		while (!inFile.eof()) {

			if (!getline(inFile, inputImagePath)) {// Get content in txt file. If eof, close the file.
				inFile.close();
				return 0;
			}
			cout << "--------------------------------------------------------------------------------------------------" << endl;
			cout<<"Processing image: "<< inputImagePath <<endl;
			Mat img = imread(inputImagePath);
			int found = inputImagePath.find_last_of("/");
			imgName = inputImagePath.substr(found + 1);

			if (img.cols < 500) resize(img, img, Size(), 2, 2);
			createWindowPartition(img, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);

			// Preprocessing
			int i = 20;
			bilateralFilter(img, imgBlur, i, i * 2, i / 2);
			cvtColor(imgBlur, imgGray, COLOR_BGR2GRAY);
			cvtColor(imgBlur, imgHSV, COLOR_BGR2HSV);
			imgResult = img.clone();

			/// License segmentation
			// HSV
			int hmin = 0, smin = 0, vmin = 0;
			int hmax = 255, smax = 255, vmax = 80;
			Scalar lower(hmin, smin, vmin);
			Scalar upper(hmax, smax, vmax);
			inRange(imgHSV, lower, upper, mask);

			maxWpixels = img.rows * img.cols * 0.6;
			// Do some preprocessing for img with much "black" noises, e.g image1.jpg
			if (countNonZero(mask)>= maxWpixels) {
				// Specify size on horizontal and veritical axis
				// use a wide and short kernel
				int horizontal_size = img.cols / 25;
				int vertical_size = img.rows / 250;
				kernel2 = getStructuringElement(MORPH_RECT, Size(horizontal_size, vertical_size));
				morphologyEx(mask, mask, MORPH_OPEN, kernel2);
			}

			// Contour
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			
			vector<Rect> boundRect(contours.size());
			int minPlateArea = (img.rows * img.cols) * 0.010; //min 1 plate area
			int maxPlateArea = (img.rows * img.cols) * 0.045; //max 1 plate area
			for (int i = 0; i < contours.size(); i++)
			{
				int area = contourArea(contours[i]);

				if (area > minPlateArea)
				{
					boundRect[i] = boundingRect(contours[i]);
					Rect roi(boundRect[i]);
					imgCrop = imgGray(roi);
					resize(imgCrop, imgCrop, Size(), 2, 2);
					threshold(imgCrop, imgCrop, 0, 255, THRESH_OTSU);
					if (area <= maxPlateArea) {

						/// Character Segmentatioon
						charSegment(imgCrop, imgCrop);
						/// Character Recognition
						api->SetImage((uchar*)imgCrop.data, imgCrop.size().width, imgCrop.size().height, imgCrop.channels(), imgCrop.step1());
						api->SetVariable("user_defined_dpi", "70");
						outText = api->GetUTF8Text();

						/// Result display
						resize(imgCrop, imgCrop, Size(), 0.5, 0.5);
						cvtColor(imgCrop, imgCrop, COLOR_GRAY2BGR);
						imgCrop.copyTo(imgResult(roi));
						// Display bounding box
						rectangle(imgResult, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 1);
						// Display the label at the top of the bounding box
						int baseLine;
						Size labelSize = getTextSize(outText, FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
						putText(imgResult, outText, Point(boundRect[i].x, boundRect[i].y + round(1.5 * labelSize.height)), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
						cout << outText << endl;
						
					}
					else {//Exceeding one plate in this segment.

						vector<Mat> split;
						vector<Rect> splitRoi;
						/// Characters set Segmentatioon
						charsSetSegment(imgCrop, split, splitRoi);
						cvtColor(imgCrop, imgCrop, COLOR_GRAY2BGR);
						for (int i = 0;i < split.size();i++) {
							/// Character Recognition
							copyMakeBorder(split[i], temp, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0, 0, 0));
							api->SetImage((uchar*)temp.data, temp.size().width, temp.size().height, temp.channels(), temp.step1());
							api->SetVariable("user_defined_dpi", "70");
							outText = api->GetUTF8Text();

							/// Result display
							cvtColor(split[i], split[i], COLOR_GRAY2BGR);
							split[i].copyTo(imgCrop(splitRoi[i]));
							//Display bounding box
							rectangle(imgCrop, splitRoi[i].tl(), splitRoi[i].br(), Scalar(0, 0, 255), 2);
							//Display the label at the top of the bounding box
							putText(imgCrop, outText, Point(splitRoi[i].x, splitRoi[i].y), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 255), 2);
							cout << outText << endl;
						}
						resize(imgCrop, imgCrop, Size(), 0.5, 0.5);
						imgCrop.copyTo(imgResult(roi));
						//Display bounding box
						rectangle(imgResult, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 1);

					}
				}
				
			}

			img.copyTo(win[0]);
			imgResult.copyTo(win[1]);
			putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
			putText(legend[1], "Result", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

			imshow("Output: " + imgName, largeWin);
			waitKey(0);
			destroyAllWindows();
		}
	}
	inFile.close();
	return 0;

}