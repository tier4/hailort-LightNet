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

namespace lightNet
{
  LightNet::LightNet()
  {
  }

  LightNet::~LightNet()
  {
  }  

  std::vector<cv::Vec3b> LightNet::getArgmaxToBgr(const unsigned char *colormap, int len)
  {
    std::vector<cv::Vec3b> argmax2bgr; 
    for (int i = 0; i < len; i++) {
      unsigned char b = colormap[3 * i + 0];
      unsigned char g = colormap[3 * i + 1];
      unsigned char r = colormap[3 * i + 2];
      cv::Vec3b color = {
	static_cast<unsigned char>((int)b),
	static_cast<unsigned char>((int)g),
	static_cast<unsigned char>((int)r)};
      argmax2bgr.push_back(color);
    }
    return argmax2bgr;
  }

  std::vector<cv::Vec3b> LightNet::getArgmaxToBgr2(const unsigned char colormap[MAX_DISTANCE][3], int len)
  {
    std::vector<cv::Vec3b> argmax2bgr; 
    for (int i = 0; i < len; i++) {
      unsigned char b = colormap[i][2];
      unsigned char g = colormap[i][1];
      unsigned char r = colormap[i][0];
      cv::Vec3b color = {
	static_cast<unsigned char>((int)b),
	static_cast<unsigned char>((int)g),
	static_cast<unsigned char>((int)r)};
      argmax2bgr.push_back(color);
    }
    return argmax2bgr;
  }  

  
  void LightNet::dequantize_tensor(float *f_data, const void *data, const hailo_vstream_info_t &info) 
  {
    hailo_format_type_t format_type = info.format.type;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;    
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;
    
    for (int y = 0; y < (int)height; y++) {      
      for (int x = 0; x < (int)width; x++) {
	float rel;
	if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	  rel =  dequantInt8(((uint8_t *)data)[y * width + x], qp_scale, qp_zp);	
	} else {
	  rel =  dequantInt16(((uint16_t *)data)[y * width + x], qp_scale, qp_zp);
	}
	f_data[y * width + x] = rel;
      }
    }
  }

  std::vector<BBoxInfo> LightNet::nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
  {
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float 
    {
      if (x1min > x2min)
	{
	  std::swap(x1min, x2min);
	  std::swap(x1max, x2max);
	}
      return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float 
    {
      float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
      float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
      float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
      float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
      float overlap2D = overlapX * overlapY;
      float u = area1 + area2 - overlap2D;
      return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
		     [](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
    std::vector<BBoxInfo> out;
    for (auto& i : binfo)
      {
	bool keep = true;
	for (auto& j : out)
	  {
	    if (keep)
	      {
		float overlap = computeIoU(i.box, j.box);
		keep = overlap <= nmsThresh;
	      }
	    else
	      break;
	  }
	if (keep) out.push_back(i);
      }
    return out;
  }

  std::vector<BBoxInfo> LightNet::nmsAllClasses(const float nmsThresh,
				      std::vector<BBoxInfo>& binfo,
				      const uint32_t numClasses)
  {
    std::vector<BBoxInfo> result;
    std::vector<std::vector<BBoxInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
      {
	splitBoxes.at(box.label).push_back(box);
      }

    for (auto& boxes : splitBoxes)
      {
	boxes = nonMaximumSuppression(nmsThresh, boxes);
	result.insert(result.end(), boxes.begin(), boxes.end());
      }

    return result;
  }

  BBox LightNet::convertBboxRes(const float& bx, const float& by, const float& bw, const float& bh,
			const uint32_t& stride_h_, const uint32_t& stride_w_, const uint32_t& netW, const uint32_t& netH)
  {
    BBox b;
    // Restore coordinates to network input resolution
    float x = bx * stride_w_;
    float y = by * stride_h_;
  
    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;
  
    b.y1 = y - bh / 2;
    b.y2 = y + bh / 2;
  
    b.x1 = clamp(b.x1, 0, netW);
    b.x2 = clamp(b.x2, 0, netW);
    b.y1 = clamp(b.y1, 0, netH);
    b.y2 = clamp(b.y2, 0, netH);

    return b;
  }

  void LightNet::addBboxProposal(const float bx, const float by, const float bw, const float bh,
			 const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb,
			 const uint32_t 	image_w, const uint32_t image_h,
			 const uint32_t 	input_w, const uint32_t input_h,		       
		       
			 std::vector<BBoxInfo>& binfo)
  {
    BBoxInfo bbi;
    bbi.box = convertBboxRes(bx, by, bw, bh, stride_h_, stride_w_, input_w, input_h);
    if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2)) {
      return;
    }
    bbi.box.x1 = ((float)bbi.box.x1 / (float)input_w)*(float)image_w;
    bbi.box.y1 = ((float)bbi.box.y1 / (float)input_h)*(float)image_h;
    bbi.box.x2 = ((float)bbi.box.x2 / (float)input_w)*(float)image_w;
    bbi.box.y2 = ((float)bbi.box.y2 / (float)input_h)*(float)image_h;

		
    bbi.label = maxIndex;
    bbi.prob = maxProb;
    bbi.classId = maxIndex;//getClassId(maxIndex);
    binfo.push_back(bbi);
  }

  std::vector<BBoxInfo> LightNet::decodeTensor(OutputVStream &output_stream, void *outputs, const float *anchors, const int image_w, const int image_h, const int input_w, const int input_h, const int numClasses, const float thresh)
  {
    auto info = output_stream.get_info();
    auto shape = info.shape;
    hailo_quant_info_t quant_info = info.quant_info;
    const int o_width = shape.width;
    const int o_height = shape.height;
    const float qp_scale = quant_info.qp_scale;
    hailo_format_type_t format_type = info.format.type;    
    const float qp_zp = quant_info.qp_zp;

    int num_anchors = 3;
    std::vector<BBoxInfo> binfo;
    float scale_x_y = 2.0;
    float offset =  0.5 * (scale_x_y-1.0);
    const int chan = (4+1+numClasses)*num_anchors;
    for (uint32_t y = 0; y < (uint32_t)o_height; ++y) {
      for (uint32_t x = 0; x < (uint32_t)o_width; ++x) {
	for (uint32_t b = 0; b < (uint32_t)num_anchors; ++b) {
	  //	const int bbindex = y * o_width + x;
	  //outputs : NHWC
	  //original : NCHW
	  const int x_index = x * chan;
	  const int y_index = y * o_width * chan;		
	  float objectness;
	  if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	    objectness = dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 4)], qp_scale, qp_zp);
	  } else {
	    objectness = dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 4)], qp_scale, qp_zp);
	  }
	  if (objectness < thresh) {
	    continue;
	  }
	  const float pw = anchors[b * 2];
	  const float ph = anchors[b * 2 + 1];
	  float bx;
	  if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	    bx  = x + scale_x_y * dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 0)] , qp_scale, qp_zp) - offset;
	  } else {
	    bx  = x + scale_x_y * dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 0)] , qp_scale, qp_zp) - offset;
	  }
	  float by;
	  if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	    by = y + scale_x_y * dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 1)] ,  qp_scale, qp_zp) - offset;
	  } else {
	    by = y + scale_x_y * dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 1)] ,  qp_scale, qp_zp) - offset;
	  }
	  float bw;
	  if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	    bw = pw * dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 2)],  qp_scale, qp_zp)  * dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 2)],  qp_scale, qp_zp) * 4;
	  } else {
	    bw = pw * dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 2)],  qp_scale, qp_zp)  * dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 2)],  qp_scale, qp_zp) * 4;
	  }
	  float bh;
	  if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	    bh  = ph * dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 3)],  qp_scale, qp_zp) * dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 3)],  qp_scale, qp_zp) * 4;
	  } else {
	    bh  = ph * dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 3)],  qp_scale, qp_zp) * dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + 3)],  qp_scale, qp_zp) * 4;
	  }

	  float maxProb = 0.0f;
	  int maxIndex = -1;	
	  for (uint32_t i = 0; i < (uint32_t)numClasses; ++i) {
	    float prob;
	    if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	      prob = dequantInt8(((uint8_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + (5 + i))],   qp_scale, qp_zp);
	    } else {
	      prob = dequantInt16(((uint16_t *)outputs)[y_index + x_index + (b * (5 + numClasses) + (5 + i))],   qp_scale, qp_zp);
	    }
	

	    if (prob > maxProb){
	      maxProb = prob;
	      maxIndex = i;
	    }
	  }
	  maxProb = objectness * maxProb;

	  if (maxProb > thresh) {
	    const uint32_t stride_h = input_h / o_height;
	    const uint32_t stride_w = input_w / o_width;
	    addBboxProposal(bx, by, bw, bh, stride_h, stride_w,  maxIndex, maxProb, image_w, image_h, input_w, input_h, binfo);
	  }
	}
      }
    }
    return binfo;
  }

  void LightNet::drawBbox(cv::Mat &img, std::vector<BBoxInfo> bboxes, const uint8_t *colormap)
  {
    for (int i = 0; i < (int)bboxes.size(); i++) {
	BBoxInfo bbi = bboxes[i];
	int id = bbi.classId;
	int x1 = bbi.box.x1;
	int y1 = bbi.box.y1;
	int x2 = bbi.box.x2;
	int y2 = bbi.box.y2;
	//    if (id==0)
	cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(colormap[3*id+0], colormap[3*id+1], colormap[3*id+2]), 2);
      }
  } 

  cv::Mat LightNet::preprocess(const cv::Mat & image,
		     const int inputH,
		     const int inputW)
  {
    cv::Mat norm_image;
    cv::resize(image, norm_image, cv::Size(inputW, inputH), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(norm_image, norm_image, cv::COLOR_BGR2RGB);
    // HailoRT do not require input normalization (tofloat, div 255.0 and NHWC2NCHW)
    //  norm_image.convertTo(norm_image, CV_32FC3);
    //  norm_image.convertTo(norm_image, CV_32F);  
    //norm_image = (norm_image) / 255.0f;
    //NHWC format
    return norm_image;
  }

  //data : NHWC
  void LightNet::getMask(cv::Mat &mask, uint8_t *data, int width, int height, const uint8_t *colormap)
  {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
	//NHW
	int id = data[width *y + x];
	mask.at<cv::Vec3b>(y, x)[0] = colormap[3 * id + 0];
	mask.at<cv::Vec3b>(y, x)[1] = colormap[3 * id + 1];
	mask.at<cv::Vec3b>(y, x)[2] = colormap[3 * id + 2];
      }
    }
  }

  void LightNet::getMaskFromArgmax2bgr(cv::Mat &mask, uint8_t *data, int width, int height, const std::vector<cv::Vec3b> &argmax2bgr)
  {

    for (int y = 0; y < height; y++) {
      int stride = width * y;
      cv::Vec3b *ptr = mask.ptr<cv::Vec3b>(y);
      
      for (int x = 0; x < width; x++) {
	    //NHW
	int id = data[stride + x];
	//mask.at<cv::Vec3b>(cv::Point(x, y)) = argmax2bgr[id];
	ptr[x] = argmax2bgr[id];
      }
    }
  }  

  void LightNet::getDepthFromArgmax2bgr(cv::Mat &depth, void *data, const hailo_vstream_info_t &info, const std::vector<cv::Vec3b> &argmax2bgr)
  {
    hailo_format_type_t format_type = info.format.type;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;    
    for (int y = height / 3; y < (int)height * 5 / 6; y++) {
      cv::Vec3b *ptr = depth.ptr<cv::Vec3b>(y);
      for (int x = 0; x < (int)width; x++) {
	//NHW     
	float rel;
	if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	  rel =  dequantInt8(((uint8_t *)data)[y * width + x], qp_scale, qp_zp);	
	} else if (format_type == HAILO_FORMAT_TYPE_UINT16){
	  rel =  dequantInt16(((uint16_t *)data)[y * width + x], qp_scale, qp_zp);
	} else {
	  rel = ((float *)data)[y * width + x];
	}	
	int distance = rel * MAX_DISTANCE;
	distance = distance >= MAX_DISTANCE ? MAX_DISTANCE-1 : distance;
	ptr[x] = argmax2bgr[distance];
      }
    }
  }
  
  void LightNet::getDepth(cv::Mat &depth, void *data, const hailo_vstream_info_t &info, const std::string format)
  {
    hailo_format_type_t format_type = info.format.type;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;    
    for (int y = height / 3; y < (int)height * 5 / 6; y++) {
      for (int x = 0; x < (int)width; x++) {
	//NHW     
	float rel;
	if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	  rel =  dequantInt8(((uint8_t *)data)[y * width + x], qp_scale, qp_zp);	
	} else if (format_type == HAILO_FORMAT_TYPE_UINT16){
	  rel =  dequantInt16(((uint16_t *)data)[y * width + x], qp_scale, qp_zp);
	} else {
	  rel = ((float *)data)[y * width + x];
	}	
	int distance = rel * MAX_DISTANCE;
	distance = distance >= MAX_DISTANCE ? MAX_DISTANCE-1 : distance;
	if (format == "jet") {
	  depth.at<cv::Vec3b>(y, x)[0] = jet_colormap[distance][0];
	  depth.at<cv::Vec3b>(y, x)[1] = jet_colormap[distance][1];
	  depth.at<cv::Vec3b>(y, x)[2] = jet_colormap[distance][2];
	} else {
	  depth.at<cv::Vec3b>(y, x)[0] = magma_colormap[distance][2];
	  depth.at<cv::Vec3b>(y, x)[1] = magma_colormap[distance][1];
	  depth.at<cv::Vec3b>(y, x)[2] = magma_colormap[distance][0];
	}	
      }
    }
  }

  void LightNet::getBackProjection(cv::Mat &bev, void *data, int im_w, int im_h, cv::Mat &seg, const hailo_vstream_info_t &info)
  {
    const float ux = im_w/2.0;
    const float uy = im_h/2.0;
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;    
    float scale_w = (float)(im_w) / (float)width;
    float scale_h = (float)(im_h) / (float)height;      
    hailo_format_type_t format_type = info.format.type;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;
    float d_scale_x = (1.0 / 1.43183880e+03) * 120.0;
    float d_scale_y = (1.0 / 1.47166278e+03) * 120.0;
    float GRID_W_40 = GRID_W / 40.0;
    float gran_h = (float)GRID_H/100.0;    
    for (int y = (int)height*5/6; y > (int)(height/3); y--) {      
      for (int x = width*0.15; x < (int)width*0.85; x++) {
	float rel;

	if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	  rel =  dequantInt8(((uint8_t *)data)[y * width + x], qp_scale, qp_zp);	
	} else if (format_type == HAILO_FORMAT_TYPE_UINT16){
	  rel =  dequantInt16(((uint16_t *)data)[y * width + x], qp_scale, qp_zp);
	} else {
	  rel = ((float *)data)[y * width + x];
	}
	float distance = 120.0 * rel;
	float yy = y * scale_h;
	float y3d = (yy - uy) * d_scale_y * rel;
	if (y3d < -2.5) {
	  continue;
	}	
	float xx = x * scale_w;
	float x3d = (xx- ux) * d_scale_x * rel;
	if (x3d > 8.0) {
	  continue;
	}
	x3d = (x3d+20.0)*GRID_W_40;	
	if (x3d > GRID_H || x3d < 0.0) {
	  continue;
	}	  
	if (distance > 0.0 && distance < 100.0) {

	  if (seg.at<cv::Vec3b>((int)yy,(int)xx)[0] == 0 && seg.at<cv::Vec3b>((int)yy,(int)xx)[1] == 0 && seg.at<cv::Vec3b>((int)yy,(int)xx)[2] == 0) {
	    continue;
	  }
	  cv::Vec3b *p_seg = seg.ptr<cv::Vec3b>((int)yy);
	  cv::Vec3b value =  p_seg[(int)xx];
	  //	  for (int b = 0; b < 4; b++) {
	  for (int b = 0; b < 1; b++) {	    
	    if ((GRID_H-(int)(distance*gran_h)-b) >= 0) {
	      cv::Vec3b *p_bev = bev.ptr<cv::Vec3b>(GRID_H-(int)(distance*gran_h)-b);
	      p_bev[(int)x3d] = value;
	    }
	  }

	}
      }	
    }
  }


   cv::Mat LightNet::getHeightmap(void *data, int im_w, int im_h, const hailo_vstream_info_t &info)
  {
    const float uy = im_h/2.0;
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC3);
    float scale_h = (float)(im_h) / (float)height;      
    hailo_format_type_t format_type = info.format.type;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;
    const float max_height = 10.0; 
    for (int y = height*5/6; y > 0; y--) {      
      for (int x = 0; x < (int)width; x++) {
	float rel;

	if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	  rel =  dequantInt8(((uint8_t *)data)[y * width + x], qp_scale, qp_zp);	
	} else {
	  rel =  dequantInt16(((uint16_t *)data)[y * width + x], qp_scale, qp_zp);
	}
	float distance = 120.0 * rel;
	float yy = (float)y * scale_h;
	float y3d = ((yy - uy) / 1.47166278e+03) * distance;

	y3d += 1.5;
	y3d = y3d < 0.0 ? 0.0 : y3d;
	y3d = y3d > max_height ? max_height : y3d;
	int value = y3d / (float)max_height * 255;
	mask.at<cv::Vec3b>(y, x)[0] = jet_colormap[value][0];
	mask.at<cv::Vec3b>(y, x)[1] = jet_colormap[value][1];
	mask.at<cv::Vec3b>(y, x)[2] = jet_colormap[value][2];	
      }	
    }
    return mask;
  }

  void LightNet::plotBBoxIntoBEV(cv::Mat &bev, std::vector<BBoxInfo> bboxes, void *data, int im_w, int im_h, const hailo_vstream_info_t &info)
  {
    const float ux = im_w/2.0;
    const float uy = im_h/2.0;
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;    
    float scale_w = (float)(im_w) / (float)width;
    float scale_h = (float)(im_h) / (float)height;      
    hailo_format_type_t format_type = info.format.type;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;    
    for (const auto &b : bboxes) {
      if (b.label > 7) {
	continue;
      }
      float x1 = b.box.x1;// * scale_w;
      float x2 = b.box.x2;// * scale_w;
      float y2 = b.box.y2-4;// * scale_h;
      float c_x = (x1 + x2)/2.0;
      float c_y = y2;//(int)(y1 + y2)/2.0;
      int x = (int)(c_x * width / (float)im_w);
      int y = (int)(c_y * height / (float)im_h);
      float rel;
      if (format_type == HAILO_FORMAT_TYPE_UINT8) {
	rel =  dequantInt8(((uint8_t *)data)[y * width + x], qp_scale, qp_zp);	
      } else {
	rel =  dequantInt16(((uint16_t *)data)[y * width + x], qp_scale, qp_zp);
      }
      float distance = 120.0 * rel;
      float xx = x * scale_w;
      float yy = y * scale_h;
      float x3d = ((xx- ux) / 1.43183880e+03) * distance;      
      float y3d = ((yy - uy) / 1.47166278e+03) * distance;      	  	
      if (x3d > 20.0) {
	continue;
      }
      x3d = (x3d+20.0)*GRID_W/40.0;
      if (x3d > GRID_H || x3d < 0.0) {
	continue;
      }	  
      if (y3d < -2.5) {
	continue;
      }
      float gran_h = (float)GRID_H/100.0;
      cv::circle(bev, cv::Point((int)x3d, GRID_H-(int)(distance*gran_h)), 4, cv::Scalar(255,255,255), -1);	
    }    
  }   
}
