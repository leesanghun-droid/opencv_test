
#최소버전 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT( TENSORFLOW_LITE )
MESSAGE ( ${CMAKE_PROJECT_NAME} )

add_compile_options(-O3 -Wall -pthread -std=c++11)
add_link_options(
-ldl 
-lpthread 
-O3 
-lrt 
#-lgstreamer-1.0 
-lglib-2.0 
-lgobject-2.0 
#-ledgetpu
)

ADD_EXECUTABLE( main main.cc inferencewrapper.cc )#camerastreamer.cc
FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES   ( main ${OpenCV_LIBS} )
include_directories     (
                                /home/lsh/tensorflow/
                                /home/lsh/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
                                /home/lsh/tensorflow/tensorflow/lite/tools/make/downloads/absl
                                /home/lsh/edgetpu/libedgetpu/
                                /usr/lib/x86_64-linux-gnu/glib-2.0/include 
                                #/usr/include/gstreamer-1.0 
                                /usr/include/glib-2.0 
                        )
TARGET_LINK_LIBRARIES   (       
                                main  
                                /home/lsh/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/libflatbuffers.a 
                                /home/lsh/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a 
                                /lib/x86_64-linux-gnu/libdl.so.2 
                                /lib/x86_64-linux-gnu/libpthread.so
                                /home/lsh/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1
                                /home/lsh/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1.0
                        )




