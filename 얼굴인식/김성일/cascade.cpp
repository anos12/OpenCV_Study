
#pragma warning(disable:4819)

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(void)
{	
	Mat src;
	src = imread("girlsday1.jpg");

	if (src.empty()) {
	cerr << "Image Load Fail..." << endl;
	return -1;
	}

	CascadeClassifier classifier("haarcascade_frontalface_default.xml");
	//CascadeClassifier classifier("haarcascade_frontalface_alt.xml");

	if (classifier.empty()) {
		cerr << "XML Load Fail..." << endl;
		return -1;
	}

	vector<Rect> faces;
	//classifier.detectMultiScale(src, faces);
	classifier.detectMultiScale(src,1.1,4);

	for (Rect rc : faces) {
		rectangle(src, rc, Scalar(255, 0, 255), 2);
	}

	imshow("Image", src);

	waitKey(0);
	destroyAllWindows();
	return 0;	
}