#include <iostream>
#include "Eigen/Dense"

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdexcept>

Eigen::Vector3d rotationAndTranslateWithBase(double angle, Eigen::Vector3d translate, Eigen::Vector3d p);

using namespace std;


//    Eigen::Vector3d p(2.0, 1.0, 1.0);
//    Eigen::Vector3d translate(1.0, 2.0, 1.0);
//
//    cout<<"result"<<endl;
//    cout<<rotationAndTranslateWithBase(45.0 * (M_PI / 180.0), translate, p) <<endl;


constexpr double MY_PI = 3.1415926;


float degree2Rapid(float angle) {
    return angle * (MY_PI / 180.0f);
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
            0, 1, 0, -eye_pos[1],
            0, 0, 1, -eye_pos[2],
            0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle, Eigen::Vector3f k)
{
    //計算出真正的角度
    float angle = degree2Rapid(rotation_angle);
    Eigen::Matrix4f model;

    Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();

    Eigen::Matrix3f Rk;

    // k x v 轉換後得到的Rk表達形式
    Rk << 0, -k[2], k[1],
        k[2], 0, -k[0],
        -k[1], k[0], 0;

    Eigen::Matrix3f M;

    M = identity + (1 - cos(angle)) * Rk * Rk + Rk * sin(angle);

    model << M(0, 0), M(0, 1), M(0, 2), 0,
            M(1, 0), M(1, 1), M(1, 2), 0,
            M(2, 0), M(2, 1), M(2, 2), 0,
            0, 0, 0, 1;

    return model;
}


Eigen::Matrix4f get_model_matrix_homogeneous (float rotation_angle, Eigen::Vector3f k)
{
    //計算出真正的角度
    float angle = degree2Rapid(rotation_angle);
    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f Rk;

    // k x v 轉換後得到的Rk表達形式  其次坐標表示形式
    Rk << 0, k[2], k[1], 0,
        k[2], 0, -k[0], 0,
        -k[1], k[0], 0, 0,
        0, 0, 0, 1;

    Eigen::Matrix4f M;

    M = identity + (1 - cos(angle)) * Rk * Rk + Rk * sin(angle);

    return M;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{


    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    float eye_angle = degree2Rapid(eye_fov);
    float height = std::tan(eye_angle / 2.0f) * std::abs(zNear) * 2.0f;
    float width = height * aspect_ratio;

    //算出 l r b t
    float t = height / 2.0f;
    float b = -t;
    float l = - width / 2.0f;
    float r = -l;


    /* 分步驟求導過程

    Eigen::Matrix4f M_pers_2_ortho;
    M_pers_2_ortho <<
        zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar,  - zNear * zFar,
        0, 0, 1, 0;

    //根據寬高比 算出l, r, b, t;

    Eigen::Matrix4f  M_ortho_scale;
    M_ortho_scale <<
        2.0f / width, 0, 0, 0,
        0, 2.0f / height, 0, 0,
        0, 0, 2.0f / (zNear - zFar), 0,
        0, 0, 0, 1;

    Eigen::Matrix4f M_ortho_tran;
    M_ortho_tran <<
        1, 0, 0, -(r + l) / 2.0f,
        0, 1, 0, -(t + b) / 2.0f,
        0, 0, 1, -(zNear + zFar) / 2.0f,
        0, 0, 0, 1;

    Eigen::Matrix4f M_ortho;

    M_ortho = M_ortho_scale * M_ortho_tran;

    return M_ortho * M_pers_2_ortho; */

    //推到後一次性寫出

    projection <<
        2.0f * zNear / (r - l), 0, (l + r) /(l - r), 0,
        0, 2.0f * zNear / (t - b), (b + t) / (b - t), 0,
        0, 0, (zNear + zFar) / (zNear - zFar), 2.0f * zNear * zFar /(zFar - zNear),
        0, 0, 1, 0;
    return projection;

}

int main(int argc, const char** argv) {

    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2},
                                     {0, 2, -2},
                                     {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        Eigen::Vector3f k(0.0f, 0.0f, 1.0f);
        r.set_model(get_model_matrix(angle, k));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        Eigen::Vector3f k(0.0f, 0.0f, 1.0f);
        r.set_model(get_model_matrix(angle, k));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}

Eigen::Vector3d rotationAndTranslateWithBase(double angle, Eigen::Vector3d translate, Eigen::Vector3d p) {
    Eigen::Matrix3d M;
    M << cos(angle), -sin(angle), translate[0],
            sin(angle), cos(angle), translate[1],
            0, 0, 1;

    cout<<"M"<<endl;
    cout<<M<<endl;

    return M * p;
}



