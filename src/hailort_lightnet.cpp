// Copyright 2023 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils.hpp"
#include "lightnet.hpp"
#include "class_timer.hpp"
#include "config_parser.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iostream> // std::cin, std::cout
#include <iomanip>  // std::setw(int), std::setfill(char)
#include <string>   // std::string
#include <sstream>  // std::stringstream
#include <omp.h>
std::vector<cv::Vec3b> argmax2bgr4semseg;
std::vector<cv::Vec3b> argmax2bgr4lane;

template<class T, class U>
bool contain(const std::basic_string<T>& s, const U& v) {
  return s.find(v) != std::basic_string<T>::npos;
}

void doPreprocessInThread(std::unique_ptr<lightNet::LightNet> lightNet, cv::VideoCapture &cap, cv::Mat &src, cv::Mat &input, int input_w, int input_h)
{
  Timer timer;
  timer.reset();
  cap >> src;
  if (src.empty() == false) {
    input = lightNet->preprocess(src, input_w, input_h);
  }
  timer.out("Preprocess");
}

void inferInThread(std::unique_ptr<hailortCommon::HrtCommon> hrtCommon, std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, unsigned char *data, std::vector<std::vector<uint8_t>> &results, float *inftime)
{
  Timer timer;
  timer.reset();
  hailo_status status = hrtCommon->infer(input_streams, output_streams, data, results);
  if (HAILO_SUCCESS != status) {
    std::cerr << "Inference failed "  << status << std::endl;
  }
  *inftime = timer.out("Inference");
}

void doPostprocessInThread(std::unique_ptr<lightNet::LightNet> lightNet, std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, std::vector<cv::Mat> &outputs, std::vector<std::vector<uint8_t>> &results , cv::Mat &src, float elapsed, float inftime, float max_power)
{
  /*PostProcessing on CPU*/
  Timer timer;
  timer.reset();
  int det_count = 0;
  auto input_info = input_streams[0].get_info();  
  const int input_w = input_info.shape.width;    
  const int input_h = input_info.shape.height;  
  std::vector<lightNet::BBoxInfo> bbox;
  std::vector<cv::Mat> segs;
  cv::Mat bev = cv::Mat::zeros(GRID_H, GRID_W, CV_8UC3);
  const int image_w = src.cols;
  const int image_h = src.rows;
  const float anchors[] = {7, 14,  17, 21,  11, 38,  31, 40,  21, 84,  57, 69,  87,130, 148,198, 215,337};
  const float nmsThresh = get_nms_thresh();
  const int numClasses = get_classes();
  const float thresh = get_score_thresh();  
  const std::string depthFormat = getDepthColorFormat();
  int depth_index = -1;

  for (int output_index = 0 ; output_index < (int)output_streams.size(); output_index++) {
    hailo_vstream_info_t info = output_streams[output_index].get_info();
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;
    auto feature = shape.features;
    //std::cout << info.name << std::endl;
    if (contain(std::string(info.name), "/argmax")) {
      //segmentation
      cv::Mat mask, resized;
      float alpha = 0.45;
      if (output_index > 0) {
	lightNet->getMaskFromArgmax2bgr(outputs[output_index], (uint8_t *)results[output_index].data(), width, height, argmax2bgr4lane);	
	alpha = 1.0;
      } else {
	lightNet->getMaskFromArgmax2bgr(outputs[output_index], (uint8_t *)results[output_index].data(), width, height, argmax2bgr4semseg);	
      }
      cv::resize(outputs[output_index], resized, cv::Size(image_w, image_h), 0, 0, cv::INTER_NEAREST);
      cv::addWeighted(src, 1.0, resized, alpha, 0.0, src);
      cv::namedWindow("mask" + std::to_string(output_index), cv::WINDOW_NORMAL);
      cv::imshow("mask"+std::to_string(output_index), outputs[output_index]);
      segs.emplace_back(resized);
    } else {
      if (feature == 1) {
	//depth estimation
	depth_index = output_index;
	std::vector<float> f_data(width * height, 0);
	lightNet->dequantize_tensor(f_data.data(), (void *)results[output_index].data(), info);
	info.format.type = HAILO_FORMAT_TYPE_FLOAT32;
#pragma omp parallel sections
	{
#pragma omp section
	  {
	    lightNet->getDepth(outputs[output_index], (void *)f_data.data(), info, depthFormat);	
	  }
#pragma omp section
	  {	  
	    for (auto &seg : segs) {
	      lightNet->getBackProjection(bev, (void *)f_data.data(), image_w, image_h, seg, info);	  
	    }
	  }
	}
	cv::namedWindow("depth" + std::to_string(output_index), cv::WINDOW_NORMAL);
	cv::imshow("depth"+std::to_string(output_index), outputs[output_index]);             
	//cv::Mat heightmap = lightNet->getHeightmap((void *)results[output_index].data(), resized.cols, resized.rows, info);
	//cv::namedWindow("heightmap" + std::to_string(output_index), cv::WINDOW_NORMAL);
	//cv::imshow("heightmap"+std::to_string(output_index), heightmap);	
      } else {
	//Object Detection
	std::vector<lightNet::BBoxInfo> b = lightNet->decodeTensor(std::ref(output_streams[output_index]), (void *)results[output_index].data(), &anchors[det_count*2*3], image_w, image_h, input_w, input_h, numClasses, thresh);
	bbox.insert(bbox.end(), b.begin(), b.end());
	det_count++;
      }
    }
  }
  //auto remaining = lightNet->nmsAllClasses(nmsThresh, bbox, numClasses);
  auto remaining = lightNet->nonMaximumSuppression(nmsThresh, bbox);
  lightNet->drawBbox(src, remaining, colormap);
  if (depth_index != -1) {
    const hailo_vstream_info_t info = output_streams[depth_index].get_info();
    lightNet->plotBBoxIntoBEV(bev, remaining, (void *)results[depth_index].data(), image_w, image_h, info);
    cv::namedWindow("bev", cv::WINDOW_NORMAL);
    cv::imshow("bev", bev);	  
  }

  float fps = 1000/elapsed;
  cv::putText(src, "FPS :" + format(fps, 4) , cv::Point(20, src.rows-60), 0, 1, cv::Scalar(255, 255,255), 1);
  cv::putText(src, "DNN Time :" + format(inftime, 4) , cv::Point(240, src.rows-60), 0, 1, cv::Scalar(255, 255,255), 1);      
  cv::putText(src, "Peak Power :" + format(max_power, 4) + "W", cv::Point(20, src.rows-20), 0, 1, cv::Scalar(255, 255,255), 1);
  cv::namedWindow("inference", cv::WINDOW_NORMAL);
  cv::imshow("inference", src);    
  if (cv::waitKey(1) == 'q');// break;
  timer.out("Postprocess");
}

int main(int argc, char* argv[])
{
  std::vector<std::vector<uint8_t>> results;
  int output_index;
  cv::VideoCapture video;
  cv::Mat src;
  std::vector<cv::Mat> outputs;    
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const std::string videoName = get_video_path();
  const int cam_id = get_camera_id();
  std::string hef_path = get_hef_path();
  auto hrtCommon = std::make_unique<hailortCommon::HrtCommon>();
  Expected<std::unique_ptr<VDevice>> vdevice = VDevice::create();  
  if (vdevice) {
    std::cerr << "Failed create vdevice, status = " << vdevice.status() << std::endl;
  }
  
  Expected<std::vector<std::reference_wrapper<Device>>> physical_devices = vdevice.value()->get_physical_devices();
  if (!physical_devices) {
    std::cerr << "Failed to get physical devices" << std::endl;
  }
  
  auto measurement_type = HAILO_POWER_MEASUREMENT_TYPES__POWER;
  auto network_group = hrtCommon->configureNetworkGroup(*vdevice.value(), hef_path);
  if (!network_group) {
    std::cerr << "Failed to configure network group " << hef_path << std::endl;
    return network_group.status();
  }

  auto vstreams = VStreamsBuilder::create_vstreams(*network_group.value(), QUANTIZED, FORMAT_TYPE);
  if (!vstreams) {
    std::cerr << "Failed creating vstreams " << vstreams.status() << std::endl;
    return vstreams.status();
  }

  if (vstreams->first.size() > MAX_LAYER_EDGES || vstreams->second.size() > MAX_LAYER_EDGES) {
    std::cerr << "Trying to infer network with too many input/output virtual streams, Maximum amount is " <<
      MAX_LAYER_EDGES << " (either change HEF or change the definition of MAX_LAYER_EDGES)"<< std::endl;
    return HAILO_INVALID_OPERATION;
  }

  for (output_index = 0 ; output_index < (int)vstreams->second.size(); output_index++) {
    auto size = vstreams->second[output_index].get_frame_size();
    std::vector<uint8_t> data(size);
    results.emplace_back(data);      
  }

  for (auto &physical_device : physical_devices.value()) {
    auto p_status = physical_device.get().stop_power_measurement();
    p_status = physical_device.get().set_power_measurement(MEASUREMENT_BUFFER_INDEX, DVM_OPTION, measurement_type);    
    p_status = physical_device.get().start_power_measurement(AVERAGE_FACTOR, SAMPLING_PERIOD);
    if (HAILO_SUCCESS != p_status) {
      std::cerr << "Failed to start measurement" << std::endl;
      return p_status;
    }    
  }

  auto input_info = vstreams->first[0].get_info();  
  const int input_w = input_info.shape.width;    
  const int input_h = input_info.shape.height;
  auto lightNet = std::make_unique<lightNet::LightNet>();
  argmax2bgr4semseg = lightNet->getArgmaxToBgr(&(semseg_colormap[0]), 20);
  argmax2bgr4lane = lightNet->getArgmaxToBgr(lane_colormap, 3);  
  float elapsed = 0.0;
  float inftime = 0.0;  

  for (int output_index = 0 ; output_index < (int)vstreams->second.size(); output_index++) {
    hailo_vstream_info_t info = vstreams->second[output_index].get_info();
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;
    cv::Mat output = cv::Mat::zeros(height, width, CV_8UC3);
    outputs.emplace_back(output);
  }
  
  if (cam_id != -1) {
    video.open(cam_id);
  } else {
    video.open(videoName);
  }
  cv::Mat input;
  video >> src;
  if (!src.empty()) {
    input = lightNet->preprocess(src, input_w, input_h);
  }
  Timer timer;
  float max_power = 0.0;      
  while (1) {
    if (src.empty() == true) break;
    timer.reset();
    cv::resize(src, src, cv::Size(1440, 960), 0, 0, cv::INTER_NEAREST);    
    cv::Mat in = input.clone();
    cv::Mat vis = src.clone();
    cv::Mat copy = src.clone();
    cv::namedWindow("img", cv::WINDOW_NORMAL);
    cv::imshow("img", copy);    
    std::vector<std::vector<uint8_t>> prev(results.begin(), results.end());
    std::thread inference_thread(inferInThread, move(hrtCommon), std::ref(vstreams->first),std::ref(vstreams->second), std::ref(in.data), std::ref(results), &inftime);
    std::thread postprocess_thread(doPostprocessInThread, move(lightNet), std::ref(vstreams->first), std::ref(vstreams->second), std::ref(outputs), std::ref(prev), std::ref(vis), elapsed, inftime, max_power);
    std::thread preprocess_thread(doPreprocessInThread, move(lightNet), std::ref(video), std::ref(src), std::ref(input), input_w, input_h);
    inference_thread.join();
    postprocess_thread.join();
    preprocess_thread.join();    
    elapsed = timer.out("Total");
    for (auto &physical_device : physical_devices.value()) {
      auto measurement_result = physical_device.get().get_power_measurement(MEASUREMENT_BUFFER_INDEX, true);
      //hrtCommon->printMeasurementsResults(physical_device.get(), measurement_result.value(), measurement_type);
      max_power = measurement_result.value().max_value;
    }    
  }
  
  for (auto &physical_device : physical_devices.value()) {  
    auto p_status = physical_device.get().stop_power_measurement();
    if (HAILO_SUCCESS != p_status) {
      std::cerr << "Failed to stop measurement" << std::endl;
      return p_status;
    }        
  }
  
  return HAILO_SUCCESS;
}
