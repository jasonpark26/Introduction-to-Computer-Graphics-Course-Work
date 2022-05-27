////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stack>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Class to store tree
////////////////////////////////////////////////////////////////////////////////
class AABBTree
{
public:
    class Node
    {
    public:
        AlignedBox3d bbox;
        int parent;   // Index of the parent node (-1 for root)
        int left;     // Index of the left child (-1 for a leaf)
        int right;    // Index of the right child (-1 for a leaf)
        int triangle; // Index of the node triangle (-1 for internal nodes)
    };

    std::vector<Node> nodes;
    int root;

    AABBTree() = default;                           // Default empty constructor
    AABBTree(const MatrixXd &V, const MatrixXi &F); // Build a BVH from an existing mesh

    int build_recursive(int start, int end, int parent, std::vector<int> &indices, const MatrixXd &centroids);
};

////////////////////////////////////////////////////////////////////////////////
// Scene setup, global variables
////////////////////////////////////////////////////////////////////////////////
const std::string data_dir = DATA_DIR;
const std::string filename("raytrace.png");
const std::string mesh_filename(data_dir + "bunny.off");

//Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; //45 degrees
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 2);

// Triangle Mesh
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)
AABBTree bvh;

//Maximum number of recursive calls
const int max_bounce = 5;

// Objects
std::vector<Vector3d> sphere_centers;
std::vector<double> sphere_radii;
std::vector<Matrix3d> parallelograms;


//Material for the object, same material for all objects
const Vector4d obj_ambient_color(0.0, 0.5, 0.0, 0);
const Vector4d obj_diffuse_color(0.5, 0.5, 0.5, 0);
const Vector4d obj_specular_color(0.2, 0.2, 0.2, 0);
const double obj_specular_exponent = 256.0;
const Vector4d obj_reflection_color(0.7, 0.7, 0.7, 0);

// Precomputed (or otherwise) gradient vectors at each grid node
const int grid_size = 20;
std::vector<std::vector<Vector2d>> grid;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;
//Ambient light
const Vector4d ambient_light(0.2, 0.2, 0.2, 0);

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
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

    grid.resize(grid_size + 1);
    for (int i = 0; i < grid_size + 1; ++i)
    {
        grid[i].resize(grid_size + 1);
        for (int j = 0; j < grid_size + 1; ++j)
            grid[i][j] = Vector2d::Random().normalized();
    }
    
    //Spheres
    sphere_centers.emplace_back(10, 0, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(7, 0.05, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(4, 0.1, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(1, 0.2, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-2, 0.4, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-5, 0.8, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-8, 1.6, 1);
    sphere_radii.emplace_back(1);

    //parallelograms
    parallelograms.emplace_back();
    parallelograms.back() << -100, 100, -100,
        -1.25, 0, -1.2,
        -100, -100, 100;


    //setup tree
    bvh = AABBTree(vertices, facets);

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);
}

////////////////////////////////////////////////////////////////////////////////
// BVH Code
////////////////////////////////////////////////////////////////////////////////

AlignedBox3d bbox_from_triangle(const Vector3d& a, const Vector3d& b, const Vector3d& c)
{
    AlignedBox3d box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}

int AABBTree::build_recursive(int start, int end, int parent, std::vector<int> &indices, const MatrixXd &centroids) {
    if (end - start == 0)
    {
        std::cout << "mesh empty" << std::endl;
        return -1;
    }
    else if (end - start == 1)
    {
        const int tid = indices[start];

        const int i0 = facets(tid, 0);
        const int i1 = facets(tid, 1);
        const int i2 = facets(tid, 2);

        const Vector3d v0 = vertices.row(i0);
        const Vector3d v1 = vertices.row(i1);
        const Vector3d v2 = vertices.row(i2);

        Node node;

        node.bbox = bbox_from_triangle(v0, v1, v2);
        node.parent = parent;   // Index of the parent node (-1 for root)
        node.left = -1;     // Index of the left child (-1 for a leaf)
        node.right = -1;    // Index of the right child (-1 for a leaf)
        node.triangle = tid; // Index of the node triangle (-1 for internal nodes)

        nodes.push_back(node);

        return nodes.size() - 1;
    }
    else
    {
        AlignedBox3d tmp_box;
        for (int i = start; i < end; i++)
        {
            Vector3d tmp = centroids.row(indices[i]);
            tmp_box.extend(tmp);
        }
        const Vector3d diag = tmp_box.diagonal();
        int max_extend = 0;
        for (int d = 0; d < 3; d++)
        {
            if (diag(d) > diag(max_extend))
                max_extend = d;
        }

        std::sort(indices.begin() + start, indices.begin() + end, [&](const int i, const int j)
        {
            return centroids(indices[i], max_extend) < centroids(indices[j], max_extend);
        });
        const int middle = (start + end) / 2;
        nodes.emplace_back();

        //Node &node = nodes.back();
        //node.parent = parent;   // Index of the parent node (-1 for root)
        const int current_node_id = nodes.size() - 1;

        const int left = build_recursive(start, middle, current_node_id, indices, centroids);
        const int right = build_recursive(middle, end, current_node_id, indices, centroids);

        //Node& node = nodes[current_node_id];
        nodes[current_node_id].parent = parent;
        nodes[current_node_id].left = left;     // Index of the left child (-1 for a leaf)
        nodes[current_node_id].right = right;    // Index of the right child (-1 for a leaf)
        nodes[current_node_id].triangle = -1; // Index of the node triangle (-1 for internal nodes)

        AlignedBox3d current_box;
        assert(left < nodes.size());
        assert(right < nodes.size());
        current_box.extend(nodes[left].bbox);
        current_box.extend(nodes[right].bbox);

        nodes[current_node_id].bbox = current_box;

        return current_node_id;

    }
}


AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F)
{
    // Compute the centroids of all the triangles in the input mesh
    MatrixXd centroids(F.rows(), V.cols());
    centroids.setZero();
    for (int i = 0; i < F.rows(); ++i)
    {
        for (int k = 0; k < F.cols(); ++k)
        {
            centroids.row(i) += V.row(F(i, k));
        }
        centroids.row(i) /= F.cols();
    }

    std::vector<int> indices(F.rows());
    for (int i = 0; i < F.rows(); ++i)
        indices[i] = i;

    root = build_recursive(0, indices.size(), -1, indices, centroids);

        //centroids.row(indices[k]);
        // TODO

        // Split each set of primitives into 2 sets of roughly equal size,
        // based on sorting the centroids along one direction or another.
}



////////////////////////////////////////////////////////////////////////////////
// Intersection code
////////////////////////////////////////////////////////////////////////////////

double ray_triangle_intersection(const Vector3d& ray_origin, const Vector3d& ray_direction, const Vector3d& a, const Vector3d& b, const Vector3d& c, Vector3d& p, Vector3d& N)
{
    // Restriction help from https ://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
    // And code from previous assignments (i.e. parallelogram intersection)

    const Vector3d pgram_origin = a;
    const Vector3d pgram_u = b - pgram_origin;
    const Vector3d pgram_v = c - pgram_origin;

    const Vector3d pd = ray_origin - pgram_origin;

    const Vector3d pv = ray_direction.cross(pgram_v);
    double determinant = pgram_u.dot(pv);

    double invDet = 1 / determinant;

    double u = pd.dot(pv) * invDet;
    if (u < 0 || u > 1) {
        return -1;
    }

    Vector3d qv = pd.cross(pgram_u);
    double v = ray_direction.dot(qv) * invDet;

    if (v < 0 || u + v > 1) {
        return -1;
    }

    Matrix3d A_tri(3, 3);
    A_tri << pgram_u, pgram_v, -ray_direction;

    Vector3d B_tri(pd(0), pd(1), pd(2));
    Vector3d x = A_tri.colPivHouseholderQr().solve(B_tri);

    double alpha = x(0);
    double beta = x(1);
    double t = x(2);

    if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1 && t >= 0))
    {
        return -1;
    }

    p = ray_origin + (t * ray_direction);
    N = (pgram_u.cross(pgram_v)).normalized();

    return t;
}

bool ray_box_intersection(const Vector3d& ray_origin, const Vector3d& ray_direction, const AlignedBox3d& box)
{
    // Code help from
    // https ://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

    float tmin = (box.min()(0) - ray_origin(0)) / ray_direction(0);
    float tmax = (box.max()(0) - ray_origin(0)) / ray_direction(0);

    if (tmin > tmax) {
        std::swap(tmin, tmax);
    }

    float tymin = (box.min()(1) - ray_origin(1)) / ray_direction(1);
    float tymax = (box.max()(1) - ray_origin(1)) / ray_direction(1);

    if (tymin > tymax) {
        std::swap(tymin, tymax);
    }

    if ((tmin > tymax) || (tymin > tmax)) {
        return false;
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    float tzmin = (box.min()(2) - ray_origin(1)) / ray_direction(2);
    float tzmax = (box.max()(2) - ray_origin(1)) / ray_direction(2);

    if (tzmin > tzmax) {
        std::swap(tzmin, tzmax);
    }

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false;
    }

    if (tzmin > tmin) {
        tmin = tzmin;
    }

    if (tzmax < tmax) {
        tmax = tzmax;
    }

    return true;
}

double ray_sphere_intersection(const Vector3d& ray_origin, const Vector3d& ray_direction, int index, Vector3d& p, Vector3d& N)
{
    //Code reused from A3
    const Vector3d sphere_center = sphere_centers[index];
    const double sphere_radius = sphere_radii[index];

    double t = -1;

    double a = ray_direction.dot(ray_direction);
    double b = 2 * ray_origin.dot(ray_direction) - 2 * ray_direction.dot(sphere_center);
    double c = (sphere_center - ray_origin).dot(sphere_center - ray_origin) - pow(sphere_radius, 2);

    double disc = pow(b, 2) - 4 * (a * c);

    if (disc < 0)
    {
        return -1;
    }
    double t1 = (-b + sqrt(disc)) / (2 * a);
    double t2 = (-b - sqrt(disc)) / (2 * a);

    t = std::min(t1, t2);
    if (t < 0) {
        t = std::max(t1, t2);
    }
    if (t < 0) {
        return -1;
    }

    p = ray_origin + (t * ray_direction);
    N = (p - sphere_center).normalized();
    return t;

}

//Compute the intersection between a ray and a paralleogram, return -1 if no intersection
double ray_parallelogram_intersection(const Vector3d& ray_origin, const Vector3d& ray_direction, int index, Vector3d& p, Vector3d& N)
{
    //Code from A3
    const Vector3d pgram_origin = parallelograms[index].col(0);
    const Vector3d A = parallelograms[index].col(1);
    const Vector3d B = parallelograms[index].col(2);
    const Vector3d pgram_u = A - pgram_origin;
    const Vector3d pgram_v = B - pgram_origin;

    const Vector3d pd = ray_origin - pgram_origin;

    Matrix3d a(3, 3);
    a << pgram_u, pgram_v, -ray_direction;

    Vector3d b(pd(0), pd(1), pd(2));
    Vector3d x = a.colPivHouseholderQr().solve(b);

    double alpha = x(0);
    double beta = x(1);
    double t = x(2);

    if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1 && t >= 0))
    {
        return -1;
    }

    p = ray_origin + (t * ray_direction);
    N = (pgram_v.cross(pgram_u)).normalized();

    return t;
}

//Finds the closest intersecting object returns its index
//In case of intersection it writes into p and N (intersection point and normals)
bool find_nearest_object(const Vector3d& ray_origin, const Vector3d& ray_direction, Vector3d& p, Vector3d& N)
{
    //Some code from A3
    Vector3d tmp_p, tmp_N;

    int closest_index = -1;
    double closest_t = std::numeric_limits<double>::max(); //closest t is "+ infinity"
    bool has_intersection = false;

    if (false)
    {
        for (int i = 0; i < facets.rows(); ++i)
        {
            // Goes through facets, then gets the 3 values needed for triangles
            // Then using the three values find the index and obtain the row values needed from vertices
            Vector3d a_facet = vertices.row(facets(i, 0)).transpose();
            Vector3d b_facet = vertices.row(facets(i, 1)).transpose();
            Vector3d c_facet = vertices.row(facets(i, 2)).transpose();

            const double t = ray_triangle_intersection(ray_origin, ray_direction, a_facet, b_facet, c_facet, tmp_p, tmp_N);

            if (t >= 0)
            {
                //The point is before our current closest t
                if (t < closest_t)
                {
                    closest_index = i;
                    closest_t = t;
                    p = tmp_p;
                    N = tmp_N;
                    has_intersection = true;
                }
            }
        }

        for (int i = 0; i < sphere_centers.size(); ++i)
        {
            const double t = ray_sphere_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
            if (t >= 0)
            {
                if (t < closest_t)
                {
                    closest_index = i;
                    closest_t = t;
                    p = tmp_p;
                    N = tmp_N;
                    has_intersection = true;
                }
            }
        }

        for (int i = 0; i < parallelograms.size(); ++i)
        {
            const double t = ray_parallelogram_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
            if (t >= 0)
            {
                if (t < closest_t)
                {
                    closest_index = sphere_centers.size() + i;
                    closest_t = t;
                    p = tmp_p;
                    N = tmp_N;
                    has_intersection = true;
                }
            }
        }
    }
    else
    {
        std::stack<int> nodes;
        nodes.push(bvh.root);

        while (!nodes.empty())
        {
            const int n_id = nodes.top();
            nodes.pop();
            const AABBTree::Node& current_node = bvh.nodes[n_id];

            if (!ray_box_intersection(ray_origin, ray_direction, current_node.bbox))
                continue;

            if (current_node.left == -1)
            {
                assert(current_node.right == -1);
                //LEAF

                const int i0 = facets(current_node.triangle, 0);
                const int i1 = facets(current_node.triangle, 1);
                const int i2 = facets(current_node.triangle, 2);

                const Vector3d v0 = vertices.row(i0);
                const Vector3d v1 = vertices.row(i1);
                const Vector3d v2 = vertices.row(i2);

                const double t = ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2, tmp_p, tmp_N);

                if (t >= 0)
                {
                    if (t < closest_t)
                    {
                        p = tmp_p;
                        N = tmp_N;
                        closest_t = t;
                        has_intersection = true;
                    }
                }
            }
            else
            {
                nodes.push(current_node.left);
                nodes.push(current_node.right);
            }
        }
    }

    /*
    if (closest_index == -1) {
        return false;
    }
    */
    return has_intersection;

    // Method (1): Traverse every triangle and return the closest hit.
    // Method (2): Traverse the BVH tree and test the intersection with a
    // triangles at the leaf nodes that intersects the input ray.
}

////////////////////////////////////////////////////////////////////////////////
// Raytracer code
////////////////////////////////////////////////////////////////////////////////
bool is_light_visible(const Vector3d& ray_origin, const Vector3d& ray_direction, const Vector3d& light_position)
{   
    //Copied from A3 with adjustments with checkinf if is_index is false or not
    Vector3d p, N;
    bool is_index;
    is_index = find_nearest_object(ray_origin, ray_direction, p, N);

    if (is_index == false) {
        return true;
    }

    long double distance = sqrt(pow((p(0) - ray_origin(0)), 2.0) + pow((p(1) - ray_origin(1)), 2.0) + pow((p(2) - ray_origin(2)), 2.0));
    long double to_light = sqrt(pow((light_position(0) - ray_origin(0)), 2.0) + pow((light_position(1) - ray_origin(1)), 2.0) + pow((light_position(2) - ray_origin(2)), 2.0));

    if (distance > to_light) {
        return true;
    }

    return false;
}


Vector4d shoot_ray(const Vector3d& ray_origin, const Vector3d& ray_direction, int max_bounce)
{
    //Intersection point and normal, these are output of find_nearest_object
    // Code reused from A3
    Vector3d p, N;

    const bool nearest_object = find_nearest_object(ray_origin, ray_direction, p, N);

    if (!nearest_object)
    {
        // Return a transparent color
        return Vector4d(0, 0, 0, 0);
    }

    // Ambient light contribution
    const Vector4d ambient_color = obj_ambient_color.array() * ambient_light.array();

    // Punctual lights contribution (direct lighting)
    Vector4d lights_color(0, 0, 0, 0);
    for (int i = 0; i < light_positions.size(); ++i)
    {
        const Vector3d& light_position = light_positions[i];
        const Vector3d& light_color = light_colors[i];

        Vector4d diff_color = obj_diffuse_color;

        // Diffuse contribution
        const Vector3d Li = (light_position - p).normalized();

        const Vector3d offset = (p + (.0001) * Li);
        if (!(is_light_visible(offset, Li, light_position))) {
            continue;
        }

        Vector3d v = (camera_position - p).normalized();
        Vector3d h = (v + Li).normalized();

        const Vector4d diffuse = diff_color * std::max(Li.dot(N), 0.0);

        // Specular contribution
        const Vector3d Hi = (Li - ray_direction).normalized();
        const Vector4d specular = obj_specular_color * std::pow(std::max(N.dot(Hi), 0.0), obj_specular_exponent);
        // Vector3d specular(0, 0, 0);

        // Attenuate lights according to the squared distance to the lights
        const Vector3d D = light_position - p;
        lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
    }

    Vector4d refl_color = obj_reflection_color;
    if (nearest_object == 4)
    {
        refl_color = Vector4d(0.5, 0.5, 0.5, 0);
    }

    // r = 2n(n·v)-v -> direction of the reflected
    Vector4d reflection_color(0, 0, 0, 0);
    if (max_bounce > 0) {
        Vector3d r = ray_direction.normalized() - 2 * N * (N.dot(ray_direction.normalized()));
        Vector3d offset_2 = p + (.0001 * r);
        Vector4d rays = shoot_ray(offset_2, r, max_bounce - 1);
        reflection_color(0) = refl_color(0) * rays(0);
        reflection_color(1) = refl_color(1) * rays(1);
        reflection_color(2) = refl_color(2) * rays(2);
        reflection_color(3) = refl_color(3) * rays(3);
    }

    // Rendering equation
    Vector4d C = ambient_color + lights_color + reflection_color;

    //Set alpha to 1
    C(3) = 1;

    return C;
}

////////////////////////////////////////////////////////////////////////////////

void raytrace_scene()
{
    std::cout << "Simple ray tracer." << std::endl;

    int w = 640;
    int h = 480;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

    // The camera always points in the direction -z
    // The sensor grid is at a distance 'focal_length' from the camera center,
    // and covers an viewing angle given by 'field_of_view'.
    double aspect_ratio = double(w) / double(h);
    //TODO
    double image_y = std::tan(field_of_view / 2.0) * focal_length;
    double image_x = image_y * aspect_ratio;

    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    const Vector3d image_origin(-image_x, image_y, camera_position[2] - focal_length);
    const Vector3d x_displacement(2.0 / w * image_x, 0, 0);
    const Vector3d y_displacement(0, -2.0 / h * image_y, 0);

    for (unsigned i = 0; i < w; ++i)
    {
        for (unsigned j = 0; j < h; ++j)
        {
            const Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;

            // Prepare the ray
            Vector3d ray_origin;
            Vector3d ray_direction;

            if (is_perspective)
            {
                // Perspective camera
                ray_origin = camera_position;
                ray_direction = (pixel_center - camera_position).normalized();
            }
            else
            {
                // Orthographic camera
                ray_origin = pixel_center;
                ray_direction = Vector3d(0, 0, -1);
            }

            const Vector4d C = shoot_ray(ray_origin, ray_direction, max_bounce);
            R(i, j) = C(0);
            G(i, j) = C(1);
            B(i, j) = C(2);
            A(i, j) = C(3);
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    setup_scene();

    raytrace_scene();
    return 0;
}
