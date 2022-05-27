// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

Vector3d intersects(Vector3d u, Vector3d v, Vector3d direction, Vector3d pgram) {
    Matrix3d a(3,3);
    a << u, v, -direction;

    Vector3d b(pgram(0),pgram(1),pgram(2));
    Vector3d x = a.colPivHouseholderQr().solve(b);
    return x;
}

void raytrace_sphere()
{
    std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

    const std::string filename("sphere_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the
    // unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = camera_origin;
            const Vector3d ray_direction = (pixel_center - camera_origin).normalized();

            // Intersect with the sphere
            const double sphere_radius = 0.9;
            const Vector3d sphere_center(0, 0, 0);

            Vector3d ec = ray_origin - sphere_center;
            double a = ray_direction.dot(ray_direction);
            double b = 2 * ray_direction.dot(ec);
            double c = ec.dot(ec) - pow(sphere_radius, 2);

            double disc = pow(b, 2) - 4 * (a * c);

            if (disc >=0)
            {
                double t = (-b - sqrt(disc)) / (2 * a);
                Vector3d ray_intersection = ray_origin + (t * ray_direction);

                // Compute normal at the intersection point
                Vector3d ray_normal = (ray_intersection - sphere_center).normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_parallelogram()
{
    std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

    const std::string filename("plane_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(0, 0.7, -10);
    const Vector3d pgram_v(1, 0.4, 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;
            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            const Vector3d p = ray_origin - pgram_origin;
            
            Vector3d x = intersects(pgram_u, pgram_v, ray_direction, p);

            double alpha = x(0);
            double beta = x(1);
            double t = x(2);

            if (alpha >= 0 && alpha <= 1 && beta >=0 && beta <= 1 && t >= 0)
            {

                Vector3d ray_intersection = pgram_origin + (alpha * pgram_u) + (beta * pgram_v);

                Vector3d ray_normal = (pgram_v.cross(pgram_u)).normalized();
                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_perspective()
{
    std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

    const std::string filename("plane_perspective.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(0, 0.7, -10);
    const Vector3d pgram_v(1, 0.4, 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = ray_origin - camera_origin;

            const Vector3d p = ray_origin - pgram_origin;
            
            Vector3d x = intersects(pgram_u, pgram_v, ray_direction, p);

            double alpha = x(0);
            double beta = x(1);
            double t = x(2);

            if (alpha >= 0 && alpha <= 1 && beta >=0 && beta <= 1 && t >= 0)
            {
                Vector3d ray_intersection = pgram_origin + (alpha * pgram_u) + (beta * pgram_v);

                Vector3d ray_normal = (pgram_v.cross(pgram_u)).normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_shading()
{
    std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

    const std::string filename("shading.png");
    
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    MatrixXd R = MatrixXd::Zero(800, 800);
    MatrixXd G = MatrixXd::Zero(800, 800);
    MatrixXd B = MatrixXd::Zero(800, 800);

    MatrixXd D = MatrixXd::Zero(800, 800);

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / A.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / A.rows(), 0);

    //Sphere setup
    const Vector3d sphere_center(0, 0, 0);
    const double sphere_radius = 0.9;

    //material params
    const Vector3d diffuse_color(1, 0, 1);
    const double specular_exponent = 100;
    const Vector3d specular_color(0., 0, 1);

    // Single light source
    const Vector3d light_position(-1, 1, 1);
    double ambient = 0.1;

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            Vector3d ec = ray_origin - sphere_center;
            double a = ray_direction.dot(ray_direction);
            double b = 2 * ray_direction.dot(ec);
            double c = ec.dot(ec) - pow(sphere_radius, 2);

            double disc = pow(b, 2) - 4 * (a * c);

            if (disc >= 0)
            {
                double t = (-b - sqrt(disc)) / (2 * a);
                Vector3d ray_intersection = ray_origin + (t * ray_direction);

                // Compute normal at the intersection point
                Vector3d ray_normal = (ray_intersection - sphere_center).normalized();

                // The light direction l (a unit vector pointing to the light source), the light_position - ray intersection
                Vector3d l = (light_position - ray_intersection).normalized();
                // The view direction v (a unit vector pointing toward the camera)
                Vector3d v = (camera_origin - ray_intersection).normalized();
                // The surface normal n (a vector perpendicular to the surface at the point of intersection)
                Vector3d n = ray_normal;
                // h is the norm of v+l
                Vector3d h = (v + l).normalized();

                const double diffuse = (std::max((n.dot(l)), 0.));
                const double specular =  (pow(std::max((n.dot(h)), 0.),specular_exponent));

                // Simple diffuse model
                C(i, j) = ambient + diffuse + specular;

                R(i, j) = ambient + (diffuse_color(0) * diffuse) + (specular_color(0) * specular);
                G(i, j) = ambient + (diffuse_color(1) * diffuse) + (specular_color(1) * specular);
                B(i, j) = ambient + (diffuse_color(2) * diffuse) + (specular_color(2) * specular);

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                R(i, j) = std::max(R(i, j), 0.);
                G(i, j) = std::max(G(i, j), 0.);
                B(i, j) = std::max(B(i, j), 0.);

                //Stronger variance
                D(i, j) = C(i, j) + 1;

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    // Write_matrix_to_uint8(R,G,B,A,image);
    write_matrix_to_png(R, G, B, A, filename);
    write_matrix_to_png(D, C, C, A, "red.png");
    write_matrix_to_png(C, D, C, A, "green.png");
    write_matrix_to_png(C, C, D, A, "blue.png");
    write_matrix_to_png(D, C, D, A, "pink.png");

}

int main()
{
    raytrace_sphere();
    raytrace_parallelogram();
    raytrace_perspective();
    raytrace_shading();

    return 0;
}
