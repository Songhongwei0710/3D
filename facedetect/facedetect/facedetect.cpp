
#include <opencv2/opencv.hpp>
#include <iostream>

#include "facedetect-dll.h"

//#pragma comment(lib,"libfacedetect.lib")
//#pragma comment(lib,"libfacedetect-x64.lib")

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000


using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{

	Mat frame;
	Mat gray;
	bool stop = true;

	VideoCapture capture(0);//打开摄像头  
	if (!capture.isOpened())
	{
		cout << "open camera error!!!" << endl;
		return -1;
	}

	int * pResults = NULL;
	//pBuffer is used in the detection functions.
	//If you call functions in multiple threads, please create one buffer for each thread!
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return -1;
	}

	int doLandmark = 1;

	while (stop)
	{
		capture >> frame;//读取当前帧到frame矩阵中  
		cvtColor(frame, gray, CV_BGR2GRAY);//转为灰度图  

										   //pResults = facedetect_frontal(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
										   //	1.2f, 2, 48, 0, doLandmark);

										   //pResults = facedetect_frontal_surveillance(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
										   //	1.2f, 2, 48, 0, doLandmark);

										   //pResults = facedetect_multiview(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
										   //	1.2f, 2, 48, 0, doLandmark);

		pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			1.2f, 3, 48, 0, doLandmark);

		printf("%d faces detected.\n", (pResults ? *pResults : 0));
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];

			printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
			rectangle(frame, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
			if (doLandmark)
			{
				for (int j = 0; j < 60; j++)
					circle(frame, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
			}
		}

		imshow("libface", frame);

		if (waitKey(30) >= 0)
			stop = false;
	}
	return 0;
}

