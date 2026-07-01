#pragma once

#ifdef _WIN32
#define YK_EXPORT extern "C" __declspec(dllexport)
#else
#define YK_EXPORT extern "C"
#endif

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
    int* hit_count_out);

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
    int* hit_count_out);

YK_EXPORT const char* yurameki_cuda_last_error();
