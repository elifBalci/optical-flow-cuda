#include <iostream>
#include <ctime>
#include <cmath>
#include "bits/time.h"

#include <opencv2/opencv.hpp>
#include <cassert>
#include <cmath>
#include <chrono>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>

#define TestCUDA true

using namespace cv;
using namespace std;

int main()
{
   Mat frame;
   Mat prevFrame;
   cuda::GpuMat prevFrameGPU;
   cuda::GpuMat frameGPU;
   VideoCapture cap("3.mp4");
   Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
   cap >> prevFrame;
   cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);
   Mat flow;

   for (;;)
   {
      cap >> frame;
      cv::resize(frame, frame, prevFrame.size());
      cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

      cuda::GpuMat prevFrameGPU(prevFrame);
      cuda::GpuMat frameGPU(frame);

      cuda::GpuMat gflow(frame.size(), CV_32FC2);
      farn->calc(prevFrameGPU, frameGPU, gflow);
      gflow.download(flow);

      for (int y = 0; y < frame.rows - 1; y += 10)
      {
         for (int x = 0; x < frame.cols - 1; x += 10)
         {
            const Point2f xyFlow = flow.at<Point2f>(y, x) * 5;
            line(frame, Point(x, y), Point(cvRound(x + xyFlow.x), cvRound(y + xyFlow.y)), Scalar(0, 255, 0), 2);
            circle(frame, Point(x, y), 1, Scalar(0, 0, 255), -1);
         }
      }

      imshow("Display window", frame);
      waitKey(25);
      frame.copyTo(prevFrame);
   }
}
