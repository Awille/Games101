// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "Eigen/Core"


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    //bounding box
    float boundingBoxLeft = std::min(v[0][0], std::min(v[1][0], v[2][0]));
    float boundingBoxRight = std::max(v[0][0], std::max(v[1][0], v[2][0]));
    float boundingBoxBottom = std::min(v[0][1], std::min(v[1][1], v[2][1]));
    float boundingBoxTop = std::max(v[0][1], std::max(v[1][1], v[2][1]));


    bool useMSAA = true;

    if (useMSAA) {
        std::vector<Eigen::Vector2f> pixel_divide
                {
                        {0.25,0.25},
                        {0.75,0.25},
                        {0.25,0.75},
                        {0.75,0.75}
                };
        for (int i = (int )floor(boundingBoxLeft); i < (int )ceil(boundingBoxRight); i ++) {
            for (int j = (int ) floor(boundingBoxBottom); j < (int ) ceil(boundingBoxTop); j ++) {
                int count = 0;
                float minDepth = FLT_MAX;
                for (int pixel = 0; pixel < 4; pixel ++) {
                    float x = i + pixel_divide[pixel][0]; float y = j + pixel_divide[pixel][1];
                    if (insideTriangle(t.v, x, y)) {
                        // If so, use the following code to get the interpolated z value.
                        auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                        float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;
                        minDepth = std::min(minDepth, z_interpolated);
                        count ++;
                    }
                }
                if (count > 0) {
                    int pixelIndex = get_index(i, j);
                    if (minDepth < depth_buf[pixelIndex]) {
                        depth_buf[pixelIndex] = minDepth;
                        frame_buf[pixelIndex] = t.getColor();
                        Eigen::Vector3f point;
                        point << i, j, minDepth;
                        set_pixel(point, t.getColor());
                    }
                }
            }
        }
        return;
    }

    for (int i = (int )floor(boundingBoxLeft); i < (int )ceil(boundingBoxRight); i ++) {
        for (int j = (int ) floor(boundingBoxBottom); j < (int ) ceil(boundingBoxTop); j ++) {
            float x = i + 0.5f; float y = j + 0.5f;
            if (insideTriangle(t.v, x, y)) {
                // If so, use the following code to get the interpolated z value.
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                int pixelIndex = get_index(i, j);
                if (z_interpolated < depth_buf[pixelIndex]) {
                    depth_buf[pixelIndex] = z_interpolated;
                    frame_buf[pixelIndex] = t.getColor();
                    Eigen::Vector3f point;
                    point << i, j, z_interpolated;
                    set_pixel(point, t.getColor());
                }
            }
        }
    }


}


/**
 * 向量叉乘（也称为向量叉积或向量外积）是一种在三维空间中定义的操作，用于计算两个三维向量之间的结果向量。
 * 向量叉乘的结果向量垂直于原始两个向量所在的平面，并且其大小与两个向量之间的夹角和两个向量的长度有关。
 * 在二维空间中，没有定义向量叉乘的概念，因为在二维空间中，两个向量总是在同一平面内，无法产生垂直于两个向量的结果向量。
 * 在二维空间中，可以使用向量的叉乘的一种特殊情况，称为 "伪叉乘" 或 "叉乘的标量形式"，其结果是一个标量而不是向量。
 * 伪叉乘的结果等于两个向量的长度乘积与其夹角的正弦值的乘积，这可以表示为以下公式：
 * 对于二维向量 u = (u1, u2) 和 v = (v1, v2)：u x v = u1 * v2 - u2 * v1
 * 注意，这里的结果是一个标量（即一个实数），而不是一个向量。这与三维空间中向量叉乘的结果不同，后者是一个向量。
 * @param pos 三角形顶点
 * @param x  点的x坐标
 * @param y  点的y坐标
 * @return
 */
bool rst::rasterizer::insideTriangle(const Eigen::Vector3f* pos, float x, float y) {
    Eigen::Vector2f ab = {pos[1][0] - pos[0][0], pos[1][1] - pos[0][1]};
    Eigen::Vector2f bc = {pos[2][0] - pos[1][0], pos[2][1] - pos[1][1]};
    Eigen::Vector2f ca = {pos[0][0] - pos[2][0], pos[0][1] - pos[2][1]};

    Eigen::Vector2f ap = {x - pos[0][0], y - pos[0][1]};
    Eigen::Vector2f bp = {x - pos[1][0], y - pos[1][1]};
    Eigen::Vector2f cp = {x - pos[2][0], y - pos[2][1]};

    float z1 = ab[0] * ap[1] - ab[1] * ap[0];
    float z2 = bc[0] * bp[1] - bc[1] * bp[0];
    float z3 = ca[0] * cp[1] - ca[1] * cp[0];

    if ((z1 >= 0 && z2 >= 0 && z3 >= 0) || (z1 < 0 && z2 < 0 && z3 < 0)) {
        return true;
    }
    return false;
// TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]



}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on