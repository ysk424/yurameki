#include "yurameki_cuda_collide.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace {

static thread_local std::string g_last_error;

struct Vec3 {
    float x, y, z;
};

__host__ __device__ Vec3 make_vec3(const float* p) {
    return Vec3{p[0], p[1], p[2]};
}

__host__ __device__ Vec3 operator+(Vec3 a, Vec3 b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vec3 operator-(Vec3 a, Vec3 b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vec3 operator*(Vec3 a, float s) {
    return Vec3{a.x * s, a.y * s, a.z * s};
}

__host__ __device__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

__host__ __device__ float length_sq(Vec3 a) {
    return dot(a, a);
}

__host__ __device__ Vec3 normalize_or(Vec3 v, Vec3 fallback) {
    float len2 = length_sq(v);
    if (len2 <= 1.0e-20f) {
        return fallback;
    }
    float inv = rsqrtf(len2);
    return v * inv;
}

__device__ float clamp01(float v) {
    return fminf(1.0f, fmaxf(0.0f, v));
}

__device__ float point_triangle_distance_sq(Vec3 p, Vec3 a, Vec3 b, Vec3 c,
                                            Vec3* normal_out) {
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    Vec3 tri_n = normalize_or(cross(ab, ac), Vec3{0.0f, 0.0f, 1.0f});

    if (d1 <= 0.0f && d2 <= 0.0f) {
        *normal_out = normalize_or(p - a, tri_n);
        return length_sq(p - a);
    }

    Vec3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        *normal_out = normalize_or(p - b, tri_n);
        return length_sq(p - b);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        Vec3 q = a + ab * v;
        *normal_out = normalize_or(p - q, tri_n);
        return length_sq(p - q);
    }

    Vec3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        *normal_out = normalize_or(p - c, tri_n);
        return length_sq(p - c);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        Vec3 q = a + ac * w;
        *normal_out = normalize_or(p - q, tri_n);
        return length_sq(p - q);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        Vec3 q = b + (c - b) * w;
        *normal_out = normalize_or(p - q, tri_n);
        return length_sq(p - q);
    }

    float dist = dot(ap, tri_n);
    *normal_out = dist >= 0.0f ? tri_n : tri_n * -1.0f;
    return dist * dist;
}

__device__ float segment_segment_distance_sq(Vec3 p1, Vec3 q1,
                                             Vec3 p2, Vec3 q2,
                                             Vec3* normal_out) {
    Vec3 d1 = q1 - p1;
    Vec3 d2 = q2 - p2;
    Vec3 r = p1 - p2;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;

    if (a <= 1.0e-20f && e <= 1.0e-20f) {
        *normal_out = normalize_or(p1 - p2, Vec3{0.0f, 0.0f, 1.0f});
        return length_sq(p1 - p2);
    }
    if (a <= 1.0e-20f) {
        t = clamp01(f / e);
    } else {
        float c = dot(d1, r);
        if (e <= 1.0e-20f) {
            s = clamp01(-c / a);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            if (denom != 0.0f) {
                s = clamp01((b * f - c * e) / denom);
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clamp01(-c / a);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clamp01((b - c) / a);
            }
        }
    }

    Vec3 c1 = p1 + d1 * s;
    Vec3 c2 = p2 + d2 * t;
    *normal_out = normalize_or(c1 - c2, Vec3{0.0f, 0.0f, 1.0f});
    return length_sq(c1 - c2);
}

__device__ bool segment_triangle_intersects(Vec3 p, Vec3 q,
                                            Vec3 a, Vec3 b, Vec3 c) {
    Vec3 dir = q - p;
    Vec3 e1 = b - a;
    Vec3 e2 = c - a;
    Vec3 h = cross(dir, e2);
    float det = dot(e1, h);
    if (fabsf(det) < 1.0e-12f) {
        return false;
    }
    float inv_det = 1.0f / det;
    Vec3 s = p - a;
    float u = inv_det * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    Vec3 qv = cross(s, e1);
    float v = inv_det * dot(dir, qv);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    float t = inv_det * dot(e2, qv);
    return t >= 0.0f && t <= 1.0f;
}

__device__ float segment_triangle_distance_sq(Vec3 p, Vec3 q,
                                              Vec3 a, Vec3 b, Vec3 c,
                                              Vec3* normal_out) {
    if (segment_triangle_intersects(p, q, a, b, c)) {
        Vec3 n = normalize_or(cross(b - a, c - a), Vec3{0.0f, 0.0f, 1.0f});
        *normal_out = n;
        return 0.0f;
    }

    Vec3 n0, n1, n2, n3, n4;
    float best = point_triangle_distance_sq(p, a, b, c, &n0);
    *normal_out = n0;
    float d = point_triangle_distance_sq(q, a, b, c, &n1);
    if (d < best) { best = d; *normal_out = n1; }
    d = segment_segment_distance_sq(p, q, a, b, &n2);
    if (d < best) { best = d; *normal_out = n2; }
    d = segment_segment_distance_sq(p, q, b, c, &n3);
    if (d < best) { best = d; *normal_out = n3; }
    d = segment_segment_distance_sq(p, q, c, a, &n4);
    if (d < best) { best = d; *normal_out = n4; }
    return best;
}

__global__ void detect_kernel(int n_capsules,
                              const float* roots,
                              const float* tips,
                              float radius,
                              int n_vertices,
                              const float* vertices,
                              int n_triangles,
                              const int* triangles,
                              int* hit_flags,
                              int* hit_triangles,
                              float* hit_distances,
                              float* hit_normals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_capsules) {
        return;
    }
    Vec3 p = make_vec3(roots + i * 3);
    Vec3 q = make_vec3(tips + i * 3);
    float radius_sq = radius * radius;
    float best_sq = 3.402823466e+38f;
    int best_tri = -1;
    Vec3 best_normal{0.0f, 0.0f, 1.0f};

    for (int tri = 0; tri < n_triangles; ++tri) {
        int ia = triangles[tri * 3 + 0];
        int ib = triangles[tri * 3 + 1];
        int ic = triangles[tri * 3 + 2];
        if (ia < 0 || ib < 0 || ic < 0 ||
            ia >= n_vertices || ib >= n_vertices || ic >= n_vertices) {
            continue;
        }
        Vec3 a = make_vec3(vertices + ia * 3);
        Vec3 b = make_vec3(vertices + ib * 3);
        Vec3 c = make_vec3(vertices + ic * 3);
        Vec3 normal;
        float d_sq = segment_triangle_distance_sq(p, q, a, b, c, &normal);
        if (d_sq < best_sq) {
            best_sq = d_sq;
            best_tri = tri;
            best_normal = normal;
        }
    }

    bool hit = best_sq <= radius_sq;
    hit_flags[i] = hit ? 1 : 0;
    hit_triangles[i] = hit ? best_tri : -1;
    hit_distances[i] = sqrtf(best_sq);
    hit_normals[i * 3 + 0] = best_normal.x;
    hit_normals[i * 3 + 1] = best_normal.y;
    hit_normals[i * 3 + 2] = best_normal.z;
}

__global__ void avoid_kernel(int n_capsules,
                             const float* roots,
                             const float* tips,
                             const float* lengths,
                             float radius,
                             float max_move,
                             int n_vertices,
                             const float* vertices,
                             int n_triangles,
                             const int* triangles,
                             float* adjusted_tips,
                             int* hit_flags,
                             int* hit_triangles,
                             float* hit_distances,
                             float* hit_normals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_capsules) {
        return;
    }
    Vec3 p = make_vec3(roots + i * 3);
    Vec3 q = make_vec3(tips + i * 3);
    float radius_sq = radius * radius;
    float best_sq = 3.402823466e+38f;
    int best_tri = -1;
    Vec3 best_normal{0.0f, 0.0f, 1.0f};

    for (int tri = 0; tri < n_triangles; ++tri) {
        int ia = triangles[tri * 3 + 0];
        int ib = triangles[tri * 3 + 1];
        int ic = triangles[tri * 3 + 2];
        if (ia < 0 || ib < 0 || ic < 0 ||
            ia >= n_vertices || ib >= n_vertices || ic >= n_vertices) {
            continue;
        }
        Vec3 a = make_vec3(vertices + ia * 3);
        Vec3 b = make_vec3(vertices + ib * 3);
        Vec3 c = make_vec3(vertices + ic * 3);
        Vec3 normal;
        float d_sq = segment_triangle_distance_sq(p, q, a, b, c, &normal);
        if (d_sq < best_sq) {
            best_sq = d_sq;
            best_tri = tri;
            best_normal = normal;
        }
    }

    float best_dist = sqrtf(best_sq);
    bool hit = best_sq <= radius_sq;
    Vec3 out_tip = q;
    if (hit) {
        float push = fminf(fmaxf(radius - best_dist, 0.0f), max_move);
        Vec3 desired_tip = q + best_normal * push;
        Vec3 fallback = normalize_or(q - p, Vec3{0.0f, 0.0f, -1.0f});
        Vec3 dir = normalize_or(desired_tip - p, fallback);
        out_tip = p + dir * lengths[i];
    }

    adjusted_tips[i * 3 + 0] = out_tip.x;
    adjusted_tips[i * 3 + 1] = out_tip.y;
    adjusted_tips[i * 3 + 2] = out_tip.z;
    hit_flags[i] = hit ? 1 : 0;
    hit_triangles[i] = hit ? best_tri : -1;
    hit_distances[i] = best_dist;
    hit_normals[i * 3 + 0] = best_normal.x;
    hit_normals[i * 3 + 1] = best_normal.y;
    hit_normals[i * 3 + 2] = best_normal.z;
}

__device__ Vec3 closest_segment_triangle_normal(Vec3 p, Vec3 q,
                                                int n_vertices,
                                                const float* vertices,
                                                int n_triangles,
                                                const int* triangles,
                                                float* best_dist_out) {
    float best_sq = 3.402823466e+38f;
    Vec3 best_normal{0.0f, 0.0f, 1.0f};
    for (int tri = 0; tri < n_triangles; ++tri) {
        int ia = triangles[tri * 3 + 0];
        int ib = triangles[tri * 3 + 1];
        int ic = triangles[tri * 3 + 2];
        if (ia < 0 || ib < 0 || ic < 0 ||
            ia >= n_vertices || ib >= n_vertices || ic >= n_vertices) {
            continue;
        }
        Vec3 a = make_vec3(vertices + ia * 3);
        Vec3 b = make_vec3(vertices + ib * 3);
        Vec3 c = make_vec3(vertices + ic * 3);
        Vec3 normal;
        float d_sq = segment_triangle_distance_sq(p, q, a, b, c, &normal);
        if (d_sq < best_sq) {
            best_sq = d_sq;
            best_normal = normal;
        }
    }
    *best_dist_out = sqrtf(best_sq);
    return best_normal;
}

__global__ void gravity_frame_kernel(int n_strands,
                                     int points_per_strand,
                                     float* points,
                                     const float* prev_roots,
                                     const float* target_roots,
                                     const float* segment_lengths,
                                     float alpha,
                                     float gravity_z,
                                     float radius,
                                     int collider_substeps,
                                     float max_move,
                                     int n_vertices,
                                     const float* vertices,
                                     int n_triangles,
                                     const int* triangles,
                                     int* hit_count) {
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= n_strands) {
        return;
    }

    int point_base = si * points_per_strand;
    Vec3 prev_root = make_vec3(prev_roots + si * 3);
    Vec3 target_root = make_vec3(target_roots + si * 3);
    Vec3 root = prev_root * (1.0f - alpha) + target_root * alpha;
    points[(point_base + 0) * 3 + 0] = root.x;
    points[(point_base + 0) * 3 + 1] = root.y;
    points[(point_base + 0) * 3 + 2] = root.z;

    Vec3 gravity{0.0f, 0.0f, gravity_z};
    for (int j = 0; j < points_per_strand - 1; ++j) {
        Vec3 p = make_vec3(points + (point_base + j) * 3);
        Vec3 old_tip = make_vec3(points + (point_base + j + 1) * 3);
        float seg_len = segment_lengths[si * (points_per_strand - 1) + j];
        Vec3 desired = old_tip + gravity;
        Vec3 fallback = normalize_or(old_tip - p, Vec3{0.0f, 0.0f, -1.0f});
        Vec3 dir = normalize_or(desired - p, fallback);
        Vec3 q = p + dir * seg_len;

        for (int cstep = 0; cstep < collider_substeps; ++cstep) {
            float best_dist = 0.0f;
            Vec3 normal = closest_segment_triangle_normal(
                p, q, n_vertices, vertices, n_triangles, triangles, &best_dist);
            if (best_dist > radius) {
                break;
            }
            atomicAdd(hit_count, 1);
            float push = fminf(fmaxf(radius - best_dist, 0.0f), max_move);
            Vec3 pushed = q + normal * push;
            Vec3 pushed_dir = normalize_or(pushed - p, fallback);
            q = p + pushed_dir * seg_len;
        }

        points[(point_base + j + 1) * 3 + 0] = q.x;
        points[(point_base + j + 1) * 3 + 1] = q.y;
        points[(point_base + j + 1) * 3 + 2] = q.z;
    }
}

bool set_error(const char* prefix, cudaError_t code) {
    g_last_error = std::string(prefix) + ": " + cudaGetErrorString(code);
    return false;
}

}  // namespace

YK_EXPORT const char* yurameki_cuda_last_error() {
    return g_last_error.c_str();
}

YK_EXPORT int yurameki_cuda_detect_capsule_mesh(
    int n_capsules,
    const float* roots_xyz,
    const float* tips_xyz,
    float radius,
    int n_vertices,
    const float* vertices_xyz,
    int n_triangles,
    const int* triangles_i32,
    int* hit_flags,
    int* hit_triangles,
    float* hit_distances,
    float* hit_normals_xyz,
    int* hit_count_out) {
    g_last_error.clear();
    if (n_capsules <= 0 || n_vertices <= 0 || n_triangles <= 0 || radius < 0.0f) {
        g_last_error = "invalid collider input shape";
        return 1;
    }

    float* d_roots = nullptr;
    float* d_tips = nullptr;
    float* d_vertices = nullptr;
    int* d_triangles = nullptr;
    int* d_hit_flags = nullptr;
    int* d_hit_triangles = nullptr;
    float* d_hit_distances = nullptr;
    float* d_hit_normals = nullptr;
    int* d_hit_count = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    auto cleanup = [&]() {
        cudaFree(d_roots);
        cudaFree(d_tips);
        cudaFree(d_vertices);
        cudaFree(d_triangles);
        cudaFree(d_hit_flags);
        cudaFree(d_hit_triangles);
        cudaFree(d_hit_distances);
        cudaFree(d_hit_normals);
        cudaFree(d_hit_count);
        cudaFree(d_temp);
    };

    cudaError_t err;
    err = cudaMalloc(&d_roots, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc roots", err); cleanup(); return 2; }
    err = cudaMalloc(&d_tips, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc tips", err); cleanup(); return 2; }
    err = cudaMalloc(&d_vertices, size_t(n_vertices) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc vertices", err); cleanup(); return 2; }
    err = cudaMalloc(&d_triangles, size_t(n_triangles) * 3 * sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc triangles", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_flags, size_t(n_capsules) * sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_flags", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_triangles, size_t(n_capsules) * sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_triangles", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_distances, size_t(n_capsules) * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_distances", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_normals, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_normals", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_count, sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_count", err); cleanup(); return 2; }

    err = cudaMemcpy(d_roots, roots_xyz, size_t(n_capsules) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy roots", err); cleanup(); return 3; }
    err = cudaMemcpy(d_tips, tips_xyz, size_t(n_capsules) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy tips", err); cleanup(); return 3; }
    err = cudaMemcpy(d_vertices, vertices_xyz, size_t(n_vertices) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy vertices", err); cleanup(); return 3; }
    err = cudaMemcpy(d_triangles, triangles_i32, size_t(n_triangles) * 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy triangles", err); cleanup(); return 3; }

    int block = 128;
    int grid = (n_capsules + block - 1) / block;
    detect_kernel<<<grid, block>>>(
        n_capsules, d_roots, d_tips, radius, n_vertices, d_vertices,
        n_triangles, d_triangles, d_hit_flags, d_hit_triangles,
        d_hit_distances, d_hit_normals);
    err = cudaGetLastError();
    if (err != cudaSuccess) { set_error("detect_kernel launch", err); cleanup(); return 4; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { set_error("detect_kernel sync", err); cleanup(); return 4; }

    cub::DeviceReduce::Sum(nullptr, temp_bytes, d_hit_flags, d_hit_count, n_capsules);
    err = cudaMalloc(&d_temp, temp_bytes);
    if (err != cudaSuccess) { set_error("cudaMalloc CUB temp", err); cleanup(); return 5; }
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_hit_flags, d_hit_count, n_capsules);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { set_error("CUB reduce sync", err); cleanup(); return 5; }

    err = cudaMemcpy(hit_flags, d_hit_flags, size_t(n_capsules) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_flags", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_triangles, d_hit_triangles, size_t(n_capsules) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_triangles", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_distances, d_hit_distances, size_t(n_capsules) * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_distances", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_normals_xyz, d_hit_normals, size_t(n_capsules) * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_normals", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_count_out, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_count", err); cleanup(); return 6; }

    cleanup();
    return 0;
}

YK_EXPORT int yurameki_cuda_avoid_capsule_mesh(
    int n_capsules,
    const float* roots_xyz,
    const float* tips_xyz,
    const float* lengths,
    float radius,
    int n_substeps,
    float max_move,
    int n_vertices,
    const float* vertices_xyz,
    int n_triangles,
    const int* triangles_i32,
    float* adjusted_tips_xyz,
    int* hit_flags,
    int* hit_triangles,
    float* hit_distances,
    float* hit_normals_xyz,
    int* hit_count_out) {
    g_last_error.clear();
    if (n_capsules <= 0 || n_vertices <= 0 || n_triangles <= 0 || radius < 0.0f ||
        n_substeps <= 0 || max_move <= 0.0f) {
        g_last_error = "invalid collider input shape";
        return 1;
    }

    float* d_roots = nullptr;
    float* d_tips = nullptr;
    float* d_next_tips = nullptr;
    float* d_lengths = nullptr;
    float* d_vertices = nullptr;
    int* d_triangles = nullptr;
    float* d_adjusted_tips = nullptr;
    int* d_hit_flags = nullptr;
    int* d_hit_triangles = nullptr;
    float* d_hit_distances = nullptr;
    float* d_hit_normals = nullptr;
    int* d_hit_count = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    auto cleanup = [&]() {
        cudaFree(d_roots);
        cudaFree(d_tips);
        cudaFree(d_next_tips);
        cudaFree(d_lengths);
        cudaFree(d_vertices);
        cudaFree(d_triangles);
        cudaFree(d_adjusted_tips);
        cudaFree(d_hit_flags);
        cudaFree(d_hit_triangles);
        cudaFree(d_hit_distances);
        cudaFree(d_hit_normals);
        cudaFree(d_hit_count);
        cudaFree(d_temp);
    };

    cudaError_t err;
    err = cudaMalloc(&d_roots, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc roots", err); cleanup(); return 2; }
    err = cudaMalloc(&d_tips, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc tips", err); cleanup(); return 2; }
    err = cudaMalloc(&d_next_tips, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc next_tips", err); cleanup(); return 2; }
    err = cudaMalloc(&d_lengths, size_t(n_capsules) * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc lengths", err); cleanup(); return 2; }
    err = cudaMalloc(&d_vertices, size_t(n_vertices) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc vertices", err); cleanup(); return 2; }
    err = cudaMalloc(&d_triangles, size_t(n_triangles) * 3 * sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc triangles", err); cleanup(); return 2; }
    err = cudaMalloc(&d_adjusted_tips, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc adjusted_tips", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_flags, size_t(n_capsules) * sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_flags", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_triangles, size_t(n_capsules) * sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_triangles", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_distances, size_t(n_capsules) * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_distances", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_normals, size_t(n_capsules) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_normals", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_count, sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc hit_count", err); cleanup(); return 2; }

    err = cudaMemcpy(d_roots, roots_xyz, size_t(n_capsules) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy roots", err); cleanup(); return 3; }
    err = cudaMemcpy(d_tips, tips_xyz, size_t(n_capsules) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy tips", err); cleanup(); return 3; }
    err = cudaMemcpy(d_lengths, lengths, size_t(n_capsules) * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy lengths", err); cleanup(); return 3; }
    err = cudaMemcpy(d_vertices, vertices_xyz, size_t(n_vertices) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy vertices", err); cleanup(); return 3; }
    err = cudaMemcpy(d_triangles, triangles_i32, size_t(n_triangles) * 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy triangles", err); cleanup(); return 3; }

    int block = 128;
    int grid = (n_capsules + block - 1) / block;
    float* current_tips = d_tips;
    float* next_tips = d_next_tips;
    for (int step = 0; step < n_substeps; ++step) {
        avoid_kernel<<<grid, block>>>(
            n_capsules, d_roots, current_tips, d_lengths, radius, max_move,
            n_vertices, d_vertices, n_triangles, d_triangles, next_tips,
            d_hit_flags, d_hit_triangles, d_hit_distances, d_hit_normals);
        err = cudaGetLastError();
        if (err != cudaSuccess) { set_error("avoid_kernel launch", err); cleanup(); return 4; }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { set_error("avoid_kernel sync", err); cleanup(); return 4; }
        float* tmp = current_tips;
        current_tips = next_tips;
        next_tips = tmp;
    }

    cub::DeviceReduce::Sum(nullptr, temp_bytes, d_hit_flags, d_hit_count, n_capsules);
    err = cudaMalloc(&d_temp, temp_bytes);
    if (err != cudaSuccess) { set_error("cudaMalloc CUB temp", err); cleanup(); return 5; }
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_hit_flags, d_hit_count, n_capsules);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { set_error("CUB reduce sync", err); cleanup(); return 5; }

    err = cudaMemcpy(adjusted_tips_xyz, current_tips, size_t(n_capsules) * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy adjusted_tips", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_flags, d_hit_flags, size_t(n_capsules) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_flags", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_triangles, d_hit_triangles, size_t(n_capsules) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_triangles", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_distances, d_hit_distances, size_t(n_capsules) * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_distances", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_normals_xyz, d_hit_normals, size_t(n_capsules) * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_normals", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_count_out, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy hit_count", err); cleanup(); return 6; }

    cleanup();
    return 0;
}

YK_EXPORT int yurameki_cuda_simulate_gravity_frame(
    int n_strands,
    int points_per_strand,
    const float* points_xyz,
    const float* prev_roots_xyz,
    const float* target_roots_xyz,
    const float* segment_lengths,
    float gravity_step_m,
    float radius,
    int n_substeps,
    int collider_substeps,
    float max_move,
    int n_vertices,
    const float* vertices_xyz,
    int n_triangles,
    const int* triangles_i32,
    float* out_points_xyz,
    int* hit_count_out) {
    g_last_error.clear();
    if (n_strands <= 0 || points_per_strand < 2 || n_vertices <= 0 ||
        n_triangles <= 0 || radius < 0.0f || n_substeps <= 0 ||
        collider_substeps <= 0 || max_move <= 0.0f) {
        g_last_error = "invalid gravity frame input shape";
        return 1;
    }

    const int n_points = n_strands * points_per_strand;
    const int n_segments = n_strands * (points_per_strand - 1);
    float* d_points = nullptr;
    float* d_prev_roots = nullptr;
    float* d_target_roots = nullptr;
    float* d_lengths = nullptr;
    float* d_vertices = nullptr;
    int* d_triangles = nullptr;
    int* d_hit_count = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_points);
        cudaFree(d_prev_roots);
        cudaFree(d_target_roots);
        cudaFree(d_lengths);
        cudaFree(d_vertices);
        cudaFree(d_triangles);
        cudaFree(d_hit_count);
    };

    cudaError_t err;
    err = cudaMalloc(&d_points, size_t(n_points) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc sim points", err); cleanup(); return 2; }
    err = cudaMalloc(&d_prev_roots, size_t(n_strands) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc prev roots", err); cleanup(); return 2; }
    err = cudaMalloc(&d_target_roots, size_t(n_strands) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc target roots", err); cleanup(); return 2; }
    err = cudaMalloc(&d_lengths, size_t(n_segments) * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc segment lengths", err); cleanup(); return 2; }
    err = cudaMalloc(&d_vertices, size_t(n_vertices) * 3 * sizeof(float));
    if (err != cudaSuccess) { set_error("cudaMalloc sim vertices", err); cleanup(); return 2; }
    err = cudaMalloc(&d_triangles, size_t(n_triangles) * 3 * sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc sim triangles", err); cleanup(); return 2; }
    err = cudaMalloc(&d_hit_count, sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMalloc sim hit count", err); cleanup(); return 2; }

    err = cudaMemcpy(d_points, points_xyz, size_t(n_points) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy sim points", err); cleanup(); return 3; }
    err = cudaMemcpy(d_prev_roots, prev_roots_xyz, size_t(n_strands) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy prev roots", err); cleanup(); return 3; }
    err = cudaMemcpy(d_target_roots, target_roots_xyz, size_t(n_strands) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy target roots", err); cleanup(); return 3; }
    err = cudaMemcpy(d_lengths, segment_lengths, size_t(n_segments) * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy segment lengths", err); cleanup(); return 3; }
    err = cudaMemcpy(d_vertices, vertices_xyz, size_t(n_vertices) * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy sim vertices", err); cleanup(); return 3; }
    err = cudaMemcpy(d_triangles, triangles_i32, size_t(n_triangles) * 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error("cudaMemcpy sim triangles", err); cleanup(); return 3; }
    err = cudaMemset(d_hit_count, 0, sizeof(int));
    if (err != cudaSuccess) { set_error("cudaMemset sim hit count", err); cleanup(); return 3; }

    int block = 128;
    int grid = (n_strands + block - 1) / block;
    for (int step = 0; step < n_substeps; ++step) {
        float alpha = float(step + 1) / float(n_substeps);
        gravity_frame_kernel<<<grid, block>>>(
            n_strands, points_per_strand, d_points, d_prev_roots, d_target_roots,
            d_lengths, alpha, -gravity_step_m / float(n_substeps), radius,
            collider_substeps, max_move, n_vertices, d_vertices, n_triangles,
            d_triangles, d_hit_count);
        err = cudaGetLastError();
        if (err != cudaSuccess) { set_error("gravity_frame_kernel launch", err); cleanup(); return 4; }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { set_error("gravity_frame_kernel sync", err); cleanup(); return 4; }
    }

    err = cudaMemcpy(out_points_xyz, d_points, size_t(n_points) * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy sim out points", err); cleanup(); return 6; }
    err = cudaMemcpy(hit_count_out, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error("cudaMemcpy sim hit count", err); cleanup(); return 6; }

    cleanup();
    return 0;
}
