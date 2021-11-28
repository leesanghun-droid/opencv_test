#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
 
using namespace cv;
using namespace std;
 
int main(int, char**){
    
    VideoCapture cap1("v4l2src device=/dev/video0 ! jpegdec ! videoconvert ! appsink");
 
    if (!cap1.isOpened()) 
    {
        printf("error\r\n");
    }
 
    Mat frame1;
    namedWindow("camera1", 1);
 
    while (1) {
        //웹캡으로부터 한 프레임을 읽어옴
        cap1.read(frame1);
        imshow("camera1", frame1);

        
        // q키를 누르면 종료
        if (waitKey(1) == 27) break;
    }
 
    return 0;
}