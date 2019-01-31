//=================================================================================================
// Copyright (c) 2016, Stefan Kohlbrecher, TU Darmstadt
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Simulation, Systems Optimization and Robotics
//       group, TU Darmstadt nor the names of its contributors may be used to
//       endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================

#include <octomap_proc_common/octomap_cv_interface.h>

//#include "ros/ros.h"

//#include <vigir_point_cloud_proc/depth_image_to_mesh_ros.h>

int main(int argc, char **argv)
{
  //ros::init(argc, argv, "scan_to_clouds_pub_node");

  //vigir_point_cloud_proc::DepthImageToMeshRos<pcl::PointXYZ> conv;
  octomap_cv_interface::OctomapCvInterface test;

  // @TODO: Use non-hardcoded path
  test.fromFile("/home/kohlbrecher/logs/obstacles_jasmine.ot");


  test.printInfo();
  
  //test.process();

  cv::Mat cv_img;
  cv::Mat free_map;
  test.retrieveHeightMap(cv_img, free_map);


  cv::namedWindow("grid",cv::WINDOW_NORMAL);
  cv::imshow("grid", cv_img);

  cv::namedWindow("free",cv::WINDOW_NORMAL);
  cv::imshow("free", free_map);

  cv::Mat inpaint_lib;
  cv_image_convert::getInpaintedImage(cv_img, inpaint_lib, -0.5, 0.5);

  cv::namedWindow("inpaint_lib",cv::WINDOW_NORMAL);
  cv::imshow("inpaint_lib", inpaint_lib);

  cv::Mat inpaint_free;
  cv_image_convert::getInpaintedImage(free_map, inpaint_free, -0.5, 0.5);

  cv::namedWindow("inpaint_free",cv::WINDOW_NORMAL);
  cv::imshow("inpaint_free", inpaint_free);

  cv::Mat diff = inpaint_lib - inpaint_free;

  cv::namedWindow("diff",cv::WINDOW_NORMAL);
  cv::imshow("diff", diff);

  cv::Mat grad_mag_img;
  cv_image_convert::getGradientMagnitudeImage(inpaint_lib,grad_mag_img);

  cv::namedWindow("grad_mag_img",cv::WINDOW_NORMAL);
  cv::imshow("grad_mag_img", grad_mag_img);


  //cv::waitKey(0);

  //cv::Mat upper_thresh;
  //cv::threshold(cv_img, upper_thresh, 2.5, 0.0, cv::THRESH_TRUNC);

  //cv::Mat lower_thresh;
  //cv::threshold(upper_thresh, lower_thresh, 0.8, 0.0, cv::THRESH_TOZERO);



  //cv::Mat inpaint_img;
  //lower_thresh.convertTo(inpaint_img, CV_8UC1);
  //inpaint_img = lower_thresh;

  cv::Mat tmp;
  cv_image_convert::getUC8ImageFromFC1(cv_img, tmp, 0, 1.5);

  //@TODO: Do not copy here for setting size
  cv::Mat mask;
  tmp.copyTo(mask);


  for (size_t i = 0; i < tmp.total(); ++i){
    if (tmp.at<uchar>(i) == 0){
      mask.at<uchar>(i) = 255;
    }else{
      mask.at<uchar>(i) = 0;
    }
  }


  cv::Mat inpainted;
  cv::inpaint(tmp, mask, inpainted, 0.0, cv::INPAINT_NS);

  int erosion_type = cv::MORPH_RECT;
  int erosion_size = 1;
  cv::Mat element = cv::getStructuringElement( erosion_type,
                                       cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       cv::Point( erosion_size, erosion_size ) );



  cv::Mat to_inpaint_erode;
  tmp.copyTo(to_inpaint_erode);

  for (size_t i = 0; i < tmp.total(); ++i){
    if (tmp.at<uchar>(i) == 0){
      to_inpaint_erode.at<uchar>(i) = 255;
    }else{
      to_inpaint_erode.at<uchar>(i) = 0;
    }
  }

  cv::Mat eroded;
  cv::erode(to_inpaint_erode, eroded, element);
  //eroded = 255 - eroded;

  if (eroded.type() != mask.type() || eroded.channels() != mask.channels() || eroded.size != mask.size)
    std::cout << "Types not equal";
  else
    std::cout << "Types equal";
  //eroded = mask;

  //for (size_t i = 0; i < tmp.total(); ++i){
  //  std::cout << (int)eroded.at<uchar>(i) << " " << (int)mask.at<uchar>(i) << "\n";
  //}

  //to_inpaint_erode = 255 - to_inpaint_erode;

  cv::Mat local_mask;

  cv::compare(mask, eroded, local_mask, cv::CMP_NE);

  cv::Mat inpainted_eroded;
  cv::inpaint(tmp, local_mask, inpainted_eroded, 0.0, cv::INPAINT_NS);


  cv::Mat inpainted_gauss_blurred;
  cv::GaussianBlur( inpainted_eroded, inpainted_gauss_blurred, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  cv::Mat dst;
  cv::Laplacian( inpainted_gauss_blurred, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT );

  cv::Mat abs_dst;
  cv::convertScaleAbs( dst, abs_dst );


  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  cv::Sobel( inpainted_gauss_blurred, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  /// Gradient Y
  cv::Sobel( inpainted_gauss_blurred, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

  cv::convertScaleAbs( grad_x, abs_grad_x );
  cv::convertScaleAbs( grad_y, abs_grad_y );

  cv::Mat grad;
  cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );



  cv::namedWindow("grid_uc8",cv::WINDOW_NORMAL);
  cv::imshow("grid_uc8", tmp);

  cv::namedWindow("mask",cv::WINDOW_NORMAL);
  cv::imshow("mask", mask);


  cv::namedWindow("inpainted",cv::WINDOW_NORMAL);
  cv::imshow("inpainted", inpainted);

  cv::namedWindow("to_inpaint_erode",cv::WINDOW_NORMAL);
  cv::imshow("to_inpaint_erode", to_inpaint_erode);

  cv::namedWindow("eroded",cv::WINDOW_NORMAL);
  cv::imshow("eroded", eroded);

  cv::namedWindow("inpainted_eroded",cv::WINDOW_NORMAL);
  cv::imshow("inpainted_eroded", inpainted_eroded);

  cv::namedWindow("inpainted_gauss_blurred",cv::WINDOW_NORMAL);
  cv::imshow("inpainted_gauss_blurred", inpainted_gauss_blurred);

  cv::namedWindow("abs_dst",cv::WINDOW_NORMAL);
  cv::imshow("abs_dst", abs_dst);

  cv::namedWindow("grad",cv::WINDOW_NORMAL);
  cv::imshow("grad", grad);

  cv::waitKey(0);

  //ros::spin();
  return 0;
}
