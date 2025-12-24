#include <rknn_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <memory>
#include <cstdio>
#include <cstdint>
#include <Eigen/Dense>
#include <chrono>



// 调试宏
#define LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)
// ====================== Eigen 类型别名（必须在函数声明前定义） ======================
using EigenMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using EigenMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using EigenVectorXi = Eigen::Vector<int, Eigen::Dynamic>;
void print_tensor_shape(const std::string& name, const int* dims, int num_dims) {
    std::cout << name << " shape = (";
    for (int i = 0; i < num_dims; ++i) {
        std::cout << dims[i];
        if (i != num_dims - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}
// FP32转FP16（关键：匹配模型输入类型）
static uint16_t float32_to_float16(float f) {
    uint32_t u = *(uint32_t*)&f;
    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp = (u >> 23) & 0xff;
    uint32_t mantissa = u & 0x7fffff;

    // FP32 -> FP16 转换逻辑
    if (exp == 0) { // 零或非规格化数
        return (uint16_t)(sign << 15);
    }
    if (exp == 0xff) { // 无穷大或NaN
        return (uint16_t)((sign << 15) | 0x7c00 | (mantissa ? 0x1ff : 0));
    }

    // 规格化数
    exp -= 127;
    if (exp > 15) { // 溢出，返回无穷大
        return (uint16_t)((sign << 15) | 0x7c00);
    }
    if (exp < -14) { // 下溢，返回零
        return (uint16_t)(sign << 15);
    }

    exp += 15;
    mantissa >>= 13; // 截断到10位尾数
    return (uint16_t)((sign << 15) | (exp << 10) | mantissa);
}

// 预处理函数
cv::Mat preprocess(const cv::Mat& bgr_img) {
    CV_Assert(bgr_img.type() == CV_8UC3);

    cv::Mat gray(bgr_img.rows, bgr_img.cols, CV_32FC1);

    for (int y = 0; y < bgr_img.rows; ++y)
    {
        const cv::Vec3b* src = bgr_img.ptr<cv::Vec3b>(y);
        float* dst = gray.ptr<float>(y);

        for (int x = 0; x < bgr_img.cols; ++x)
        {
            // OpenCV: BGR
            float B = src[x][0];
            float G = src[x][1];
            float R = src[x][2];

            // Python: image[..., ::-1] => RGB
            dst[x] = (0.299f * R +
                      0.587f * G +
                      0.114f * B) / 255.0f;
        }
    }

    return gray; // HxW, float32
}

// 加载模型
static unsigned char* load_model(const char* model_path, int* model_size) {
    if (!model_path || !model_size) {
        LOG_ERROR("load_model: invalid params");
        return nullptr;
    }
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open model file: %s", model_path);
        return nullptr;
    }
    
    *model_size = (int)file.tellg();
    if (*model_size <= 0) {
        LOG_ERROR("Model file is empty");
        file.close();
        return nullptr;
    }
    
    unsigned char* model_data = new (std::nothrow) unsigned char[*model_size];
    if (!model_data) {
        LOG_ERROR("Failed to allocate memory for model (size: %d)", *model_size);
        file.close();
        return nullptr;
    }
    
    file.seekg(0, std::ios::beg);
    file.read((char*)model_data, *model_size);
    if (file.gcount() != *model_size) {
        LOG_ERROR("Read model file incomplete: %d/%d", file.gcount(), *model_size);
        delete[] model_data;
        file.close();
        return nullptr;
    }
    file.close();
    
    return model_data;
}

// 打印张量属性
static void print_tensor_attr(const rknn_tensor_attr* attr) {
    if (!attr) return;
    LOG_DEBUG("  index: %d", attr->index);
    LOG_DEBUG("  name: %s", attr->name);
    LOG_DEBUG("  n_dims: %d", attr->n_dims);
    LOG_DEBUG("  dims: [");
    for (uint32_t i = 0; i < attr->n_dims; ++i) {
        LOG_DEBUG("    %d%s", attr->dims[i], (i == attr->n_dims-1 ? "" : ","));
    }
    LOG_DEBUG("  ]");
    LOG_DEBUG("  fmt: %d (NHWC=%d, NCHW=%d)", attr->fmt, RKNN_TENSOR_NHWC, RKNN_TENSOR_NCHW);
    LOG_DEBUG("  type: %d (FLOAT32=%d, FLOAT16=%d)", attr->type, RKNN_TENSOR_FLOAT32, RKNN_TENSOR_FLOAT16);
    LOG_DEBUG("  size: %d", attr->size);
}


bool fill_batch2_input(const cv::Mat& img0, const cv::Mat& img1, 
                      void* out_buf, const rknn_tensor_attr& attr) {
    CV_Assert(attr.dims[0] == 2 && img0.size() == img1.size() && img0.type() == CV_32FC1);
    const int B=2, H=attr.dims[1], W=attr.dims[2], C=attr.dims[3];
    const size_t elem_size = (attr.type == RKNN_TENSOR_FLOAT16) ? 2 : 4;
    const size_t img_bytes = H * W * C * elem_size; // 单张图的字节数

    // 遍历2个batch，填充数据
    for (int b = 0; b < 2; ++b) {
        const cv::Mat& img = (b == 0) ? img0 : img1;
        uint8_t* batch_ptr = (uint8_t*)out_buf + b * img_bytes; // 当前batch的起始地址

        for (int y = 0; y < H; ++y) {
            const float* img_row = img.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                float val = img_row[x];
                // 计算像素偏移（NHWC：y*W*C + x*C；NCHW：0*H*W + y*W + x）
                size_t pixel_offset = (attr.fmt == RKNN_TENSOR_NHWC) 
                    ? (y * W + x) * C * elem_size 
                    : (0 * H + y) * W * elem_size + x * elem_size;

                if (attr.type == RKNN_TENSOR_FLOAT16) {
                    ((uint16_t*)(batch_ptr + pixel_offset))[0] = float32_to_float16(val);
                } else {
                    ((float*)(batch_ptr + pixel_offset))[0] = val;
                }
            }
        }
    }
    return true;
}
// 输出Mat第一行前N个值的辅助函数
void printFirstRowTopN(const cv::Mat& mat, int N = 100) {
    CV_Assert(mat.type() == CV_32FC1); // 确保是float32单通道
    CV_Assert(mat.rows >= 1);          // 确保有至少一行

    // 实际输出的数量（避免图片宽度不足100导致越界）
    int output_num = std::min(N, mat.cols);
    
    std::cout << "灰度图第一行前" << output_num << "个值（float32）：" << std::endl;

    // 获取第一行的float指针
    const float* first_row = mat.ptr<float>(0);
    for (int x = 0; x < output_num; ++x) {
        // 每10个值换行，方便查看
        if (x % 10 == 0 && x != 0) {
            std::cout << std::endl;
        }
        // 输出当前值（带索引）
        std::cout << "[" << x << "]:" << first_row[x] << "\t";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // 1. 参数检查
    if (argc != 3) {
        LOG_ERROR("Usage: %s <rknn_model_path> <output_img_path>", argv[0]);
        LOG_ERROR("Example: %s ./lightclue.rknn ./out.jpg", argv[0]);
        return -1;
    }
    const char* rknn_model_path = argv[1];
    const char* output_img_path = argv[2];

    // 2. 图片加载与预处理
    const std::string left_img_path = "/userdata/luoshiyong/lightglue_infer/IMG_0702.jpg";
    const std::string right_img_path = "/userdata/luoshiyong/lightglue_infer/IMG_0702.jpg"; //"/userdata/luoshiyong/lightglue_infer/cut_jtport.tif";
    cv::Mat left_img = cv::imread(left_img_path);
    cv::Mat right_img = cv::imread(right_img_path);
   

    // 3. RKNN初始化
    rknn_context ctx = 0;
    int model_size = 0;
    unsigned char* model_data = load_model(rknn_model_path, &model_size);
    if (!model_data) {
        return -1;
    }

    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    delete[] model_data;
    if (ret < 0) {
        LOG_ERROR("rknn_init failed! ret=%d", ret);
        return ret;
    }
    LOG_DEBUG("rknn_init success, ctx=0x%lx", (unsigned long)ctx);

    // 4. 查询输入属性（关键：匹配类型和size）
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_query(IN_OUT_NUM) failed! ret=%d", ret);
        rknn_destroy(ctx);
        return ret;
    }
    LOG_DEBUG("Model Input Num: %u, Output Num: %u", io_num.n_input, io_num.n_output);

    if (io_num.n_input != 1) {
        LOG_ERROR("Model input num is not 1 (actual: %u)", io_num.n_input);
        rknn_destroy(ctx);
        return -1;
    }

    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_query(INPUT_ATTR) failed! ret=%d", ret);
        rknn_destroy(ctx);
        return ret;
    }
    LOG_DEBUG("=== Input Tensor Attr ===");
    print_tensor_attr(&input_attr);

    // 解析输入维度
    int batch = input_attr.dims[0];
    int height = 0, width = 0, channel = 0;
    if (input_attr.fmt == RKNN_TENSOR_NHWC) {
        std::cout<<">>>>>>>>>>> type = NHWC"<<std::endl;
        height = input_attr.dims[1];  // H 1024
        width = input_attr.dims[2];   // W  1024
        channel = input_attr.dims[3];  //C   1
    } else if (input_attr.fmt == RKNN_TENSOR_NCHW) {
        std::cout<<">>>>>>>>>>> type = NCHW"<<std::endl;
        channel = input_attr.dims[1];
        height = input_attr.dims[2];
        width = input_attr.dims[3];
    }
    else
        std::cout<<">>>>>>>>>>> type unknown>>>>>>>>>>>>>>>>"<<std::endl;
    height = 1024;  // H 1024
    width = 1024;   // W  1024
    channel = 1;  //C   1
    LOG_DEBUG("Model input shape: batch=%d, h=%d, w=%d, c=%d", batch, height, width, channel);
    LOG_DEBUG("Model input type: %d, size: %d bytes", input_attr.type, input_attr.size);
     // 1. resize
    cv::resize(left_img, left_img,  cv::Size(width, height));
    cv::resize(right_img, right_img, cv::Size(width, height));

    // 2. preprocess（完全对齐 Python）
    cv::Mat left_gray  = preprocess(left_img);
    cv::Mat right_gray = preprocess(right_img);
    printFirstRowTopN(left_gray);
    // 3. 分配输入 buffer（你原来的代码 OK）
    std::unique_ptr<uint8_t[]> input_buf(
        new uint8_t[input_attr.size]
    );

    // 4. 填充
    fill_batch2_input(
        left_gray,
        right_gray,
        input_buf.get(),
        input_attr
    );

    // 5. 设置输入
    rknn_input input{};
    input.index = 0;
    input.buf   = input_buf.get();
    input.size  = input_attr.size;
    input.type  = input_attr.type;
    //input.fmt   = input_attr.fmt;
    input.fmt = RKNN_TENSOR_NCHW;  // 或 RKNN_TENSOR_NCHW
    //input.pass_through = 1;

    rknn_inputs_set(ctx, 1, &input);
  

    LOG_DEBUG("rknn_inputs_set success");

    // 7. 执行推理
    LOG_DEBUG("Running model...");
    auto time1  = std::chrono::high_resolution_clock::now();
    ret = rknn_run(ctx, nullptr);
    auto time2  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> load_duration = time2 - time1;
    std::cout << "model infer  Time: " << load_duration.count() << " ms" << std::endl;
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_run failed! ret=%d", ret);
        rknn_destroy(ctx);
        return ret;
    }
    LOG_DEBUG("rknn_run success");
    // ========== 关键：查询输出张量的真实属性 ==========
    std::vector<rknn_tensor_attr> output_attrs(3);
    for (uint32_t i = 0; i < 3; ++i) {
        memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG_ERROR("rknn_query(OUTPUT_ATTR[%u]) failed! ret=%d", i, ret);
            // 释放资源逻辑...
            return -1;
        }
        LOG_DEBUG("=== Output %u Real Attr ===", i);
        print_tensor_attr(&output_attrs[i]); // 打印真实类型/维度
    }

    // 8. 获取输出（保护逻辑）
    if (io_num.n_output < 3) {
        LOG_ERROR("Model output num < 3 (actual: %u)", io_num.n_output);
        rknn_destroy(ctx);
        return -1;
    }

    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    bool output_ok = true;
    // 2. 统一设置输出参数（转为FP32）
    for (uint32_t i = 0; i < 3; ++i) {
        outputs[i].index = i;      // 必须设置index
        outputs[i].want_float = 1; // 输出转为FP32
    }

    // 3. 一次性获取所有3个输出（关键！start_index=0，num=3）
    ret = rknn_outputs_get(ctx, 3, outputs, nullptr);

    if (ret != RKNN_SUCC) {
        output_ok = false;
        LOG_ERROR("rknn_outputs_get failed! ret=%d", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    // 4. 打印输出信息（验证）
    for (uint32_t i = 0; i < 3; ++i) {
        if (!outputs[i].buf) {
            LOG_ERROR("Output %u buf is null!", i);
            output_ok = false;
            break;
        }
        LOG_DEBUG("Output %u: size=%zu, buf=0x%lx", 
                  i, outputs[i].size, (unsigned long)outputs[i].buf);
    }
    if (!output_ok) {
        for (uint32_t i = 0; i < 3; ++i) {
            if (outputs[i].buf) {
                rknn_outputs_release(ctx, 1, &outputs[i]);
            }
        }
        rknn_destroy(ctx);
        return -1;
    }
    
    // 9. 提取输出数据
    float* keypoints = (float*)outputs[0].buf;
    float* matches = (float*)outputs[1].buf;
    float* mscores = (float*)outputs[2].buf;
    for (int lsy = 0; lsy<50;lsy++)
    {
        std::cout<<mscores[lsy]<<" ";
    }
     for(int idx1 = 0; idx1<512;idx1++)
    {
        float x0 = keypoints[idx1*  2 + 0];
        float y0 = keypoints[idx1 * 2 + 1];
        std::cout<<"( "<<x0<<", "<<y0<<" )"<<std::endl;

    }
    /*
    if (!keypoints || !matches || !mscores) {
        LOG_ERROR("Output buf is null!");
        rknn_outputs_release(ctx, 3, outputs);
        rknn_destroy(ctx);
        return -1;
    }

   
    // ======================= 解析匹配点（正确版） =======================
    std::vector<cv::Point2f> pts_left, pts_right;
    //keypoint  output_attrs[0] [2,512,2]
    //matchs    output_attrs[0] [50,3]
    //mscores   output_attrs[0] [50]
    int max_matches = output_attrs[1].dims[0]; // 50
    for(int idx1 = 0; idx1<512;idx1++)
    {
        float x0 = keypoints[idx1*  2 + 0];
        float y0 = keypoints[idx1 * 2 + 1];
        std::cout<<"( "<<x0<<", "<<y0<<" )"<<std::endl;

    }
    for(int idx1 = 0; idx1<512;idx1++)
    {
        float x1 = keypoints[2 * 512 + idx1*  2 + 0];
        float y1 = keypoints[2 * 512 + idx1 * 2 + 1];
        std::cout<<"( "<<x1<<", "<<y1<<" )"<<std::endl;

    }
    
    for (int i = 0; i < max_matches; ++i) {
        int batch_id = static_cast<int>(matches[i * 3 + 0]);
        int idx0     = static_cast<int>(matches[i * 3 + 1]);
        int idx1     = static_cast<int>(matches[i * 3 + 2]);

        // 通常 batch_id 恒为 0，这里还是做个保护
        //if (batch_id != 0) continue;
        if (idx0 < 0 || idx1 < 0) continue;
        if (idx0 >= 512 || idx1 >= 512) continue;
        //    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)

        float x0 = keypoints[idx0 * 2 + 0];
        float y0 = keypoints[idx0 * 2 + 1];
        float x1 = keypoints[2 * 512 + idx1*  2 + 0];
        float y1 = keypoints[2 * 512 + idx1 * 2 + 1];



        std::cout<<"( "<<x0<<", "<<y0<<" )"<<std::endl;
        std::cout<<"( "<<x1<<", "<<y1<<" )"<<std::endl;
        pts_left.emplace_back(x0, y0);
        pts_right.emplace_back(x1, y1);
    }

    LOG_DEBUG("Valid matches: %zu", pts_left.size());
    */
    

    // 12. 释放资源
    rknn_outputs_release(ctx, 3, outputs);
    rknn_destroy(ctx);
    LOG_DEBUG("All resources released");
    /*
    // ======================= 画匹配结果 =======================
    cv::Mat match_vis;
    cv::hconcat(left_img, right_img, match_vis);

    int offset_x = left_img.cols;

    for (size_t i = 0; i < pts_left.size(); ++i) {
        cv::Point2f p0 = pts_left[i];
        cv::Point2f p1 = pts_right[i];
        p1.x += offset_x;

        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
        cv::circle(match_vis, p0, 3, color, -1);
        cv::circle(match_vis, p1, 3, color, -1);
        cv::line(match_vis, p0, p1, color, 1);
    }

    cv::imwrite("matches.jpg", match_vis);
    LOG_DEBUG("Saved matches.jpg");*/

    return 0;
}