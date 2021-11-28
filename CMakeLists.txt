
#최소버전 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT( TENSORFLOW_LITE )
MESSAGE ( ${CMAKE_PROJECT_NAME} )

add_compile_options(-O3 -Wall -pthread -std=c++11)
add_link_options()
#-O3 
#-lrt 
#-ldl 
#-lpthread 
#-lgstreamer-1.0 
#-lglib-2.0 
#-lgobject-2.0 
#-ledgetpu)

ADD_EXECUTABLE( main main.cc ) #camerastreamer.cc inferencewrapper.cc )
FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES   ( main ${OpenCV_LIBS} )
include_directories     (
                                #/home/pi/tensorflow
                                #/home/pi/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
                                #/home/pi/tensorflow/tensorflow/lite/tools/make/downloads/absl
                                #/home/pi/edgetpu/libedgetpu/
                                #/usr/lib/arm-linux-gnueabihf/glib-2.0/include 
                                #/usr/include/gstreamer-1.0 
                                #/usr/include/glib-2.0 
                        )
TARGET_LINK_LIBRARIES   (       
                                main  
                                #/lib/arm-linux-gnueabihf/libdl.so.2 
                                #/home/pi/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a 
                                #/home/pi/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a 
                                #/home/pi/edgetpu/libedgetpu/direct/armv7a/libedgetpu.so.1
                                #/home/pi/edgetpu/libedgetpu/direct/armv7a/libedgetpu.so.1.0
                        )




