#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>

#include "tensorflow/lite/interpreter.h"
#include "edgetpu_c.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                      \
  }

 
using namespace cv;
using namespace std;
Mat resize_img(200,200,CV_8UC3);
std::unique_ptr<tflite::Interpreter> interpreter_;


void init()
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = tflite::FlatBufferModel::BuildFromFile("AGE.tflite");
    TFLITE_MINIMAL_CHECK(model != nullptr);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter_);

    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
    TFLITE_MINIMAL_CHECK(num_devices);
    printf("device num : %d\r\n",(int)num_devices);
    const auto &device = devices.get()[0];
    auto *delegate =
        edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    interpreter_->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});
    interpreter_->SetNumThreads(1);
    TFLITE_MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
}

 
int main(int, char**){

    init();
    
    VideoCapture cap1("v4l2src device=/dev/video0 ! jpegdec ! videoconvert ! appsink");
 
    if (!cap1.isOpened()) 
    {
        printf("error\r\n");
    }
 
    Mat frame1;
    namedWindow("camera1", 1);

    const int h =200;
    const int w =200;

    static uint8_t byte_array[sizeof(uint8_t)*h*w*3];

            //resize_img = imread("test_img/20.jpg", 1);
    
    while (1) {
        //웹캡으로부터 한 프레임을 읽어옴
        cap1.read(frame1);
        resize(frame1, resize_img, Size(200, 200));

        for(int y=0;y<h;y++)
        for(int x=0;x<w;x++)
        {
            byte_array[y*h*3+x*3+2]=resize_img.at<cv::Vec3b>(y, x)[0];
            byte_array[y*h*3+x*3+1]=resize_img.at<cv::Vec3b>(y, x)[1];
            byte_array[y*h*3+x*3+0]=resize_img.at<cv::Vec3b>(y, x)[2];
        }

        std::vector<float> output_data;
        uint8_t *input = interpreter_->typed_input_tensor<uint8_t>(0);
        std::memcpy(input, byte_array, h*w*3);

        TFLITE_MINIMAL_CHECK(interpreter_->Invoke() == kTfLiteOk);
    
        const auto &output_indices = interpreter_->outputs();
        const auto *out_tensor = interpreter_->tensor(output_indices[0]);
        TFLITE_MINIMAL_CHECK(out_tensor != nullptr);

        if (out_tensor->type == kTfLiteUInt8)
        {
            //std::cerr << "kTfLiteUInt8" << std::endl;
            const uint8_t *output1 = interpreter_->typed_output_tensor<uint8_t>(0);
            const uint8_t *output2 = interpreter_->typed_output_tensor<uint8_t>(1);
            double low = output1[0]*0.3961;
            double high = output2[0]*0.3961;
            printf("output1[0] : %d ,low : %f \r\n",output1[0],low);
            printf("output2[0] : %d ,high : %f \r\n",output2[0],high);
        }



        imshow("camera1", resize_img);

        
        // q키를 누르면 종료
        if (waitKey(1) == 27) break;
    }
 
    return 0;
}