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

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.hpp"
#include "colormap.hpp"

namespace lightNet
{
  /**
   * @class LightNet
   * @brief LightNet Inference
   */

  struct BBox
  {
    float x1, y1, x2, y2;
  };

  struct BBoxInfo
  {
    BBox box;
    int label;
    int classId;
    float prob;
  };
  
  class LightNet
  {
  public:
    LightNet();    
    
    ~LightNet();


    std::vector<cv::Vec3b>  getArgmaxToBgr(const unsigned char *colormap, int len);
    std::vector<cv::Vec3b>  getArgmaxToBgr2(const unsigned char colormap[MAX_DISTANCE][3], int len);    
    
    void dequantize_tensor(float *f_data, const void *data, const hailo_vstream_info_t &info) ;    
    

    /**
     * @brief run NMS
     * @param[in] nmsThresh threshould for NMS
     * @param[in] binfo information for bounding boxes
     * @param[in] numClasses number of classess
     * @param[out] results information for bounding boxes 
     */                    
    std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh,
					std::vector<BBoxInfo>& binfo,
					const uint32_t numClasses);

    /**
     * @brief convert bbox for yolo format
     * @param[in] bx x0
     * @param[in] by y0
     * @param[in] bw width
     * @param[in] bh height
     * @param[in] stride_h_ scale value for DNN width
     * @param[in] stride_w_ scale value for DNN height
     * @param[in] netW width for DNN inputs
     * @param[in] netH height for DNN inputs
     * @param[out] b BBox 
     */                        
    BBox convertBboxRes(const float& bx, const float& by, const float& bw, const float& bh,
				    const uint32_t& stride_h_, const uint32_t& stride_w_, const uint32_t& netW, const uint32_t& netH);
    
    /**
     * @brief decode yolo detections
     * @param[in] output_stream hailo output stream
     * @param[in] output output datas
     * @param[in] anchor anchor datas for yolo detections
     * @param[in] image_w width for an input image
     * @param[in] image_h height for an input image
     * @param[in] input_w width for DNN inputs
     * @param[in] input_h height for DNN inputs
     * @param[in] numClasses number of classes
     * @param[in] thresh threshould for detections
     * @param[out] binfo final bbox 
     */                        
    std::vector<BBoxInfo> decodeTensor(OutputVStream &output_stream, void *outputs, const float *anchors, const int image_w, const int image_h, const int input_w, const int input_h, const int numClasses, const float thresh);

    /**
     * @brief draw BBox
     * @param[in] img an input image
     * @param[in] bboxes bboxes
     * @param[in] colormap colormap for rendering
     */                            
    void drawBbox(cv::Mat &img, std::vector<BBoxInfo> bboxes, const uint8_t *colormap);

    /**
     * @brief run preprcocess
     * @param[in] image an input image
     * @param[in] inputH height for DNN inputs
     * @param[in] inputW width for DNN inputs
     * @param[out] norm_image DNN input tensor
     */                                
    cv::Mat preprocess(const cv::Mat & image,
				 const int inputH,
				 const int inputW);

    /**
     * @brief get mask for semantic segmentation
     * @param[in] data INT8 data
     * @param[in] width width for data
     * @param[in] height height for data
     * @param[in] colormap colormap for rendering
     * @param[out] mask mask image
     */                                    
    void getMask(cv::Mat &mask, uint8_t *data, int width, int height, const uint8_t *colormap);

    void getMaskFromArgmax2bgr(cv::Mat &mask, uint8_t *data, int width, int height, const std::vector<cv::Vec3b> &argmax2bgr);

    /**
     * @brief get depth for depth estimation
     * @param[in] data INT8/INT32 data
     * @param[in] width width for data
     * @param[in] height height for data
     * @param[in] info Hailo output stream information
     * @param[in] format Color format for depth visualization
     * @param[out] depth depth image
     */

    void getDepthFromArgmax2bgr(cv::Mat &depth, void *data, const hailo_vstream_info_t &info, const std::vector<cv::Vec3b> &argmax2bgr);
    
    void getDepth(cv::Mat &depth, void *data, const hailo_vstream_info_t &info, const std::string format);

    /**
     * @brief get Bird View Eye (BEB) outputs rendering with semantic segmentation
     * @param[in] data INT8/INT32 data
     * @param[in] im_w  width for data
     * @param[in] im_h height for data
     * @param[in] seg semseg image
     * @param[in] info Hailo output stream information
     * @param[out] bev BEV image
     */                                            
    void getBackProjection(cv::Mat &bev, void *data, int im_w, int im_h, cv::Mat &seg, const hailo_vstream_info_t &info);
    //void getBackProjection(cv::Mat &bev, void *data, int im_w, int im_h, std::vector<cv::Mat> &segs, const hailo_vstream_info_t &info);    
    /**
     * @brief get heightmap from depth
     * @param[in] data INT8/INT32 data
     * @param[in] im_w  width for data
     * @param[in] im_h height for data
     * @param[in] seg semseg image
     * @param[in] info Hailo output stream information
     * @param[out] bev BEV image
     */                                            
    cv::Mat getHeightmap(void *data, int im_w, int im_h, const hailo_vstream_info_t &info);

    /**
     * @brief Plot circle into bev from bboxes
     * @param[in] bev BEV image
     * @param[in] bboxes BBoxes
     * @param[in] data INT8/INT32 data
     * @param[in] im_w  width for data
     * @param[in] im_h height for data
     * @param[in] info Hailo output stream information
     */                                                
    void plotBBoxIntoBEV(cv::Mat &bev, std::vector<BBoxInfo> bboxes, void *data, int im_w, int im_h, const hailo_vstream_info_t &info);
    
    /**
     * @brief apply NMS
     * @param[in] nmsThresh threshould for NMS
     * @param[in] binfo Bboxes
     * @param[out] out final Bboxes
     */                                                
    std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo);


    /**
     * @brief add a bbox into infos for bboxes
     * @param[in] bx x0
     * @param[in] by y0
     * @param[in] bw width
     * @param[in] bh height
     * @param[in] stride_h_ scale value for DNN width
     * @param[in] stride_w_ scale value for DNN height
     * @param[in] maxIndex index
     * @param[in] maxProb probabilty
     * @param[in] image_w width for image inputs
     * @param[in] image_h height for image inputs
     * @param[in] input_w width for DNN inputs
     * @param[in] input_h height for DNN inputs
     * @param[in] binfo infos for Bboxes
     */                                                    
    void addBboxProposal(const float bx, const float by, const float bw, const float bh,
				     const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb,
				     const uint32_t 	image_w, const uint32_t image_h,
				     const uint32_t 	input_w, const uint32_t input_h,		       				     
				     std::vector<BBoxInfo>& binfo);
    
  protected:    
  private:
  };    

  #define GRID_H 600
  #define GRID_W 400
  //#define GRID_H 250
  //#define GRID_W 200    
}
