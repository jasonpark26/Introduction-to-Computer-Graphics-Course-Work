// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "raster.h"

#include <gif.h>
#include <fstream>

#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

//Image height
const int H = 480;

//Camera settings
const double near_plane = 1.5; //AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; //45 degrees
const double aspect_ratio = 1.5;
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

//Object
const std::string data_dir = DATA_DIR;
const std::string mesh_filename(data_dir + "bunny.off");
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)

//Material for the object
const Vector3f obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3f obj_specular_color(0.2, 0.2, 0.2);
const float obj_specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
//Ambient light
const Vector4f ambient_light(0.3, 0.3, 0.3, 0.0);

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    if (!in.good())
    {
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

void build_uniform(UniformAttributes &uniform)
{
    float n = -near_plane; //near plane
    float f = -far_plane; //far plane
    float t = near_plane * std::tan(field_of_view / 2); // top plane - using from a3/a4
    float r = t * aspect_ratio; // right plane - using from a3/a4
    float b = -t; //bottom plane
    float l = -r; //left plane
    
    //Calculated using lecture slides
    Vector3d w_cam = -camera_gaze.normalized();
    Vector3d u_cam = (camera_top.cross(w_cam)).normalized();
    Vector3d v_cam = w_cam.cross(u_cam);
    Vector3d e_cam = camera_position;

    Matrix4f M_cam;
    M_cam << u_cam[0], v_cam[0], w_cam[0], e_cam[0],
        u_cam[1], v_cam[1], w_cam[1], e_cam[1],
        u_cam[2], v_cam[2], w_cam[2], e_cam[2],
        0, 0, 0, 1;


    Matrix4f M_orth;
    //Used the matrix from slides
    M_orth << 2/(r-l), 0, 0, -(r+l)/(r-l),
        0, 2/(t-b), 0, -(t+b)/(t-b),
        0, 0, 2/(n-f), -(n+f)/(n-f),
        0, 0, 0, 1;
    Matrix4f P;
    if (is_perspective)
    {
        //Used the matrix from slides
        P <<n, 0, 0, 0,
            0, n, 0, 0,
            0, 0, n+f, -f*n,
            0, 0, 1, 0;
        uniform.view = M_orth * P * M_cam.inverse();
    }
    else
    {
        uniform.view = M_orth * M_cam.inverse();
    }
}

void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //Used from extras
        VertexAttributes out;
        out.position = uniform.view * va.position;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;
    
    //Using vertices and facets, push_back points of the triangle
    for (int i = 0; i < facets.rows(); i++) {
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2)));
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2)));
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2)));

    }

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

Matrix4f compute_rotation(const double alpha)
{
    //https ://en.wikipedia.org/wiki/Rotation_matrix
    //Used the rotation matrix rotating around the y axis
    Matrix4f res;
    res << std::cos(alpha), 0, std::sin(alpha), 0,
        0, 1, 0, 0,
        -std::sin(alpha), 0, std::cos(alpha), 0,
        0, 0, 0, 1;

    return res;
}

void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{

    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    Matrix4f trafo = compute_rotation(alpha);
    uniform.view = uniform.view * trafo;
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //Used from extras
        VertexAttributes out;
        out.position = uniform.view * va.position;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    //Like in main_lines from extra, instead of push_back'ing the points, push_back edge-by-edge
    //i.e. if i want to pushback edge a-b, push_back a then b, if i want to pushback edge b-c, push_back b then c...
    for (int i = 0; i < facets.rows(); i++) {
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2)));
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2)));
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2)));
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2)));
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2)));
        vertex_attributes.push_back(VertexAttributes(vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2)));
    }

    rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);
}

void get_shading_program(Program &program)
{
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        VertexAttributes out;
        out.position = uniform.view * va.position;
        const Vector4f ambient_color = ambient_light.array();
        Vector3f lights_color(0, 0, 0);
        Vector4f camera_position_4f;
        camera_position_4f << camera_position[0], camera_position[1], camera_position[2], 0;

        //Reusing the lighting composition from previous assignments, I set a lot of the variables 
        //to either 3f or 4f, and then for p and N that were replaced I use out.position and va.normal likewise

        for (int i = 0; i < light_positions.size(); ++i)
        {
            const Vector3d& light_position = light_positions[i];
            Vector4f light_position_4f;
            light_position_4f << light_position[0], light_position[1], light_position[2], 0;

            const Vector3d& light_color = light_colors[i];
            Vector3f light_color_3f;
            light_color_3f << light_color[0], light_color[1], light_color[2];

            Vector3f diff_color = obj_diffuse_color;

            const Vector4f Li = (light_position_4f - out.position).normalized();

            Vector4f v = (camera_position_4f - out.position).normalized();
            Vector4f h = (v + Li).normalized();
        
            const Vector3f diffuse = diff_color * std::max(Li.dot(va.normal), float(0.0));
            const Vector3f specular = obj_specular_color * std::pow(std::max(va.normal.dot(h), float(0.0)), obj_specular_exponent);

            const Vector4f D = light_position_4f - out.position;
            lights_color += (diffuse + specular).cwiseProduct(light_color_3f) / D.squaredNorm();
        }
        Vector4f lights_color_4f;
        lights_color_4f << lights_color(0), lights_color(1), lights_color(2), 0;
        Vector4f C = ambient_color + lights_color_4f;
        
        // after calculating light, add wtih ambient_color and set alpha mask to 0, then set out.color

        C(3) = 1;
        out.color = C;

        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //Instead of using uniform.color, use va.color that was calculated
        FragmentAttributes out(va.color(0), va.color(1), va.color(2), va.color(3));
        out.position = va.position;
        //Not sure why but when in perspective, bunny is flipped so set Z value to negatiive
        out.position[2] = -va.position[2];
        return out;

    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        // instead of just fa.position[2], subtract camera_position with fa.position to get proper z value and then use fa.position[2]
        // used depth_checker from main_depth
        Vector4f camera_position_4f;
        camera_position_4f << camera_position[0], camera_position[1], camera_position[2], 0;
        Vector4f compare;
        compare = (camera_position_4f - fa.position).normalized();

        if (compare[2] < previous.depth)
        {
            FrameBufferAttributes out(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
            //set depth to compare[2] instead of fa.position[2]
            out.depth = compare[2];
            return out;
        }
        else
            return previous;
    };
}

void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);
    Eigen::Matrix4f trafo = compute_rotation(alpha);
    uniform.view = uniform.view * trafo;
    std::vector<VertexAttributes> vertex_attributes;


    for (int i = 0; i < facets.rows(); i++) {
        //Copied the math from lecture of calculating normals
        //Create VertexAttributes for the three points
        VertexAttributes a1 = VertexAttributes(vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2));
        VertexAttributes b1 = VertexAttributes(vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2));
        VertexAttributes c1 = VertexAttributes(vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2));

        //Create Vectors from the points
        Vector3f a, b, c;
        a << vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2);
        b << vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2);
        c << vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2);

        //Calculate normals and cast it to 4f 
        Vector3f e0 = b - a;
        Vector3f e1 = c - a;
        Vector3f norm = (e0.cross(e1)).normalized();
        Vector4f normal_4f;
        normal_4f << norm[0], norm[1], norm[2], 0;

        a1.normal = normal_4f;
        b1.normal = normal_4f;
        c1.normal = normal_4f;

        //After setting normals, push_back into the vector
        vertex_attributes.push_back(a1);
        vertex_attributes.push_back(b1);
        vertex_attributes.push_back(c1);
    }

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);

    Eigen::Matrix4f trafo = compute_rotation(alpha);
    uniform.view = uniform.view * trafo;

    std::vector<VertexAttributes> vertex_attributes;

    //Create a matrix to hold the normals of the vectors
    MatrixXf vector_norms(facets.rows(), vertices.cols());
    vector_norms.setZero();

    for (int i = 0; i < vector_norms.rows(); i++) {
        Vector3f a, b, c;
        a << vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2);
        b << vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2);
        c << vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2);

        //Calculate normals
        Vector3f e0 = b - a;
        Vector3f e1 = c - a;
        Vector3f norm = (e0.cross(e1)).normalized();
        Vector4f normal_4f;
        normal_4f << norm[0], norm[1], norm[2], 0;

        //Then add to each point of a vector accordingly to the matrix
        vector_norms((facets(i, 0)), 0) += norm[0];
        vector_norms((facets(i, 0)), 1) += norm[1];
        vector_norms((facets(i, 0)), 2) += norm[2];

        vector_norms((facets(i, 1)), 0) += norm[0];
        vector_norms((facets(i, 1)), 1) += norm[1];
        vector_norms((facets(i, 1)), 2) += norm[2];

        vector_norms((facets(i, 2)), 0) += norm[0];
        vector_norms((facets(i, 2)), 1) += norm[1];
        vector_norms((facets(i, 2)), 2) += norm[2];

    }

    for (int i = 0; i < facets.rows(); i++) {
        VertexAttributes a1 = VertexAttributes(vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2));
        VertexAttributes b1 = VertexAttributes(vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2));
        VertexAttributes c1 = VertexAttributes(vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2));

        float length = vector_norms.rows();

        Vector4f a_norm, b_norm, c_norm;

        a_norm << vector_norms(facets(i, 0), 0) / length, vector_norms(facets(i, 0), 1) / length, vector_norms(facets(i, 0), 2) / length, 0;
        b_norm << vector_norms(facets(i, 1), 0) / length, vector_norms(facets(i, 1), 1) / length, vector_norms(facets(i, 1), 2) / length, 0;
        c_norm << vector_norms(facets(i, 2), 0) / length, vector_norms(facets(i, 2), 1) / length, vector_norms(facets(i, 2), 2) / length, 0;


        a1.normal = a_norm.normalized();
        b1.normal = b_norm.normalized();
        c1.normal = c_norm.normalized();
        //Same concept as flat_shading but when creating normals divide them by length of the rows to average them out
        //Then normalize and push_back

        vertex_attributes.push_back(a1);
        vertex_attributes.push_back(b1);
        vertex_attributes.push_back(c1);
    }

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;

    simple_render(frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());

    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());

    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer.setConstant(FrameBufferAttributes());

    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    //Same conceptually as main_animation
    const char* fileName = "wire.gif";
    const char* fileName1 = "flat.gif";
    const char* fileName2 = "pv.gif";
    int delay = 25;
    GifWriter g;
    GifBegin(&g, fileName, frameBuffer.rows(), frameBuffer.cols(), delay);

    float alpha = 0;
    for (float i = 0; i < 1; i += 0.05)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        wireframe_render(alpha, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
        alpha += EIGEN_PI/10;
    }

    GifEnd(&g);
    
    GifBegin(&g, fileName1, frameBuffer.rows(), frameBuffer.cols(), delay);

    alpha = 0;
    for (float i = 0; i < 1; i += 0.05)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        flat_shading(alpha, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
        alpha += EIGEN_PI / 10;
    }

    GifEnd(&g);

    GifBegin(&g, fileName2, frameBuffer.rows(), frameBuffer.cols(), delay);

    alpha = 0;
    for (float i = 0; i < 1; i += 0.05)
    {
        frameBuffer.setConstant(FrameBufferAttributes());
        pv_shading(alpha, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
        alpha += EIGEN_PI / 10;
    }

    GifEnd(&g);

    return 0;
}
