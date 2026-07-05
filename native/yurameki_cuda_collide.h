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
    int* hit_count_out);

YK_EXPORT int yurameki_cuda_simulate_chain_frame(
    int n_strands,
    int points_per_strand,
    const float* points_xyz,
    const float* prev_roots_xyz,
    const float* target_roots_xyz,
    const float* segment_lengths,
    float gravity_step_m,
    float propagation_length_m,
    float radius,
    int n_substeps,
    int collider_substeps,
    float max_move,
    int n_vertices,
    const float* vertices_xyz,
    int n_triangles,
    const int* triangles_i32,
    float* out_points_xyz,
    int* hit_count_out);

YK_EXPORT int yurameki_cuda_simulate_chain_frame_v2(
    int n_strands,
    int points_per_strand,
    const float* points_xyz,
    const float* prev_roots_xyz,
    const float* target_roots_xyz,
    const float* segment_lengths,
    float gravity_step_m,
    float propagation_length_m,
    const float* style_targets_xyz,
    const float* style_weights,
    float style_strength,
    float radius,
    int n_substeps,
    int collider_substeps,
    float max_move,
    int n_vertices,
    const float* vertices_xyz,
    int n_triangles,
    const int* triangles_i32,
    float* out_points_xyz,
    int* hit_count_out);

YK_EXPORT const char* yurameki_cuda_last_error();
