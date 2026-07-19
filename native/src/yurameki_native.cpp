#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace yurameki {

constexpr float kEps = 1.0e-8f;
constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    Vec3& operator-=(const Vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, float s) { return a *= s; }
inline Vec3 operator*(float s, Vec3 a) { return a *= s; }
inline Vec3 operator/(const Vec3& a, float s) { return a * (1.0f / s); }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float length_sq(const Vec3& a) { return dot(a, a); }
inline float length(const Vec3& a) { return std::sqrt(length_sq(a)); }
inline float& component(Vec3& value,int index) { return index==0?value.x:(index==1?value.y:value.z); }
inline Vec3 normalize_or(const Vec3& a, const Vec3& fallback = {0.0f, 0.0f, -1.0f}) {
    const float n = length(a);
    if (n > kEps) return a / n;
    const float fn = length(fallback);
    return fn > kEps ? fallback / fn : Vec3{0.0f, 0.0f, -1.0f};
}
inline Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a*(1.0f-t) + b*t; }
inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
inline Vec3 slerp_direction(const Vec3& a_,const Vec3& b_,float t) {
    const Vec3 a=normalize_or(a_),b=normalize_or(b_,a);const float alpha=clampf(t,0.0f,1.0f);
    const float c=clampf(dot(a,b),-1.0f,1.0f);
    if(c>0.9995f)return normalize_or(lerp(a,b,alpha),a);
    if(c<-0.9995f){Vec3 axis=cross(a,{1,0,0});if(length_sq(axis)<1.0e-8f)axis=cross(a,{0,1,0});axis=normalize_or(axis,{1,0,0});const float theta=kPi*alpha;return normalize_or(a*std::cos(theta)+cross(axis,a)*std::sin(theta),a);}
    const float theta=std::acos(c),sin_theta=std::sin(theta);return normalize_or(a*(std::sin((1.0f-alpha)*theta)/sin_theta)+b*(std::sin(alpha*theta)/sin_theta),a);
}

struct Quat {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

inline Quat operator-(const Quat& q) { return {-q.x, -q.y, -q.z, -q.w}; }
inline Quat conjugate(const Quat& q) { return {-q.x, -q.y, -q.z, q.w}; }
inline Quat operator*(const Quat& a, const Quat& b) {
    return {
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    };
}
inline Quat normalized(const Quat& q) {
    const float n = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (n <= kEps) return {};
    return {q.x/n, q.y/n, q.z/n, q.w/n};
}
inline Vec3 rotate(const Quat& q, const Vec3& v) {
    const Vec3 qv{q.x, q.y, q.z};
    const Vec3 t = 2.0f * cross(qv, v);
    return v + q.w*t + cross(qv, t);
}
inline Quat exp_quat(const Vec3& omega) {
    const float theta = length(omega);
    if (theta < 1.0e-8f) return normalized({0.5f*omega.x, 0.5f*omega.y, 0.5f*omega.z, 1.0f});
    const float half = 0.5f*theta;
    const float s = std::sin(half)/theta;
    return {omega.x*s, omega.y*s, omega.z*s, std::cos(half)};
}
inline Quat relative_darboux(const Quat& a, const Quat& b) {
    Quat d = conjugate(a) * b;
    return d.w < 0.0f ? -d : d;
}

struct Mat3 {
    float m[3][3]{};
    static Mat3 identity() {
        Mat3 a;
        a.m[0][0] = a.m[1][1] = a.m[2][2] = 1.0f;
        return a;
    }
};
inline Mat3 operator+(Mat3 a, const Mat3& b) {
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) a.m[r][c] += b.m[r][c];
    return a;
}
inline Mat3 operator*(Mat3 a, float s) {
    for (auto& row : a.m) for (float& v : row) v *= s;
    return a;
}
inline Vec3 operator*(const Mat3& a, const Vec3& v) {
    return {
        a.m[0][0]*v.x + a.m[0][1]*v.y + a.m[0][2]*v.z,
        a.m[1][0]*v.x + a.m[1][1]*v.y + a.m[1][2]*v.z,
        a.m[2][0]*v.x + a.m[2][1]*v.y + a.m[2][2]*v.z,
    };
}
inline float determinant(const Mat3& a) {
    return a.m[0][0]*(a.m[1][1]*a.m[2][2]-a.m[1][2]*a.m[2][1])
         - a.m[0][1]*(a.m[1][0]*a.m[2][2]-a.m[1][2]*a.m[2][0])
         + a.m[0][2]*(a.m[1][0]*a.m[2][1]-a.m[1][1]*a.m[2][0]);
}
inline Mat3 inverse(const Mat3& a) {
    const float d = determinant(a);
    if (std::abs(d) < 1.0e-20f) return {};
    const float s = 1.0f/d;
    Mat3 o;
    o.m[0][0]=(a.m[1][1]*a.m[2][2]-a.m[1][2]*a.m[2][1])*s;
    o.m[0][1]=(a.m[0][2]*a.m[2][1]-a.m[0][1]*a.m[2][2])*s;
    o.m[0][2]=(a.m[0][1]*a.m[1][2]-a.m[0][2]*a.m[1][1])*s;
    o.m[1][0]=(a.m[1][2]*a.m[2][0]-a.m[1][0]*a.m[2][2])*s;
    o.m[1][1]=(a.m[0][0]*a.m[2][2]-a.m[0][2]*a.m[2][0])*s;
    o.m[1][2]=(a.m[0][2]*a.m[1][0]-a.m[0][0]*a.m[1][2])*s;
    o.m[2][0]=(a.m[1][0]*a.m[2][1]-a.m[1][1]*a.m[2][0])*s;
    o.m[2][1]=(a.m[0][1]*a.m[2][0]-a.m[0][0]*a.m[2][1])*s;
    o.m[2][2]=(a.m[0][0]*a.m[1][1]-a.m[0][1]*a.m[1][0])*s;
    return o;
}

inline Quat shortest_arc(const Vec3& a_, const Vec3& b_) {
    const Vec3 a = normalize_or(a_, {0,0,1});
    const Vec3 b = normalize_or(b_, a);
    const float c = clampf(dot(a,b), -1.0f, 1.0f);
    if (c > 1.0f-1.0e-7f) return {};
    if (c < -1.0f+1.0e-7f) {
        Vec3 axis = cross(a,{1,0,0});
        if (length_sq(axis)<1.0e-10f) axis=cross(a,{0,1,0});
        axis=normalize_or(axis,{1,0,0});
        return {axis.x,axis.y,axis.z,0};
    }
    const Vec3 axis=cross(a,b);
    return normalized({axis.x,axis.y,axis.z,1.0f+c});
}

struct Aabb {
    Vec3 lo{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()};
    Vec3 hi{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()};
    void grow(const Vec3& p) {
        lo.x=std::min(lo.x,p.x); lo.y=std::min(lo.y,p.y); lo.z=std::min(lo.z,p.z);
        hi.x=std::max(hi.x,p.x); hi.y=std::max(hi.y,p.y); hi.z=std::max(hi.z,p.z);
    }
    void grow(const Aabb& b) { grow(b.lo); grow(b.hi); }
};
inline float distance_sq(const Aabb& b,const Vec3& p) {
    float d=0.0f;
    const float pv[3]{p.x,p.y,p.z}, lo[3]{b.lo.x,b.lo.y,b.lo.z}, hi[3]{b.hi.x,b.hi.y,b.hi.z};
    for(int k=0;k<3;++k){const float q=pv[k]<lo[k]?lo[k]-pv[k]:(pv[k]>hi[k]?pv[k]-hi[k]:0.0f); d+=q*q;}
    return d;
}
inline bool segment_aabb(const Aabb& b,const Vec3& a,const Vec3& c,float max_t=1.0f) {
    const Vec3 d=c-a;
    float t0=0.0f,t1=max_t;
    const float av[3]{a.x,a.y,a.z}, dv[3]{d.x,d.y,d.z}, lo[3]{b.lo.x,b.lo.y,b.lo.z}, hi[3]{b.hi.x,b.hi.y,b.hi.z};
    for(int k=0;k<3;++k){
        if(std::abs(dv[k])<1.0e-12f){if(av[k]<lo[k]||av[k]>hi[k]) return false; continue;}
        float x0=(lo[k]-av[k])/dv[k], x1=(hi[k]-av[k])/dv[k];
        if(x0>x1) std::swap(x0,x1); t0=std::max(t0,x0); t1=std::min(t1,x1); if(t0>t1) return false;
    }
    return true;
}

struct Triangle { Vec3 a,b,c,n,centroid; Aabb box; };
struct Hit { bool found=false; float t=1.0f; Vec3 p{},n{}; int tri=-1; };
struct Nearest { bool found=false; float distance_sq=std::numeric_limits<float>::infinity(); Vec3 p{},n{}; int tri=-1; };

inline Vec3 closest_triangle(const Vec3& p,const Triangle& t) {
    const Vec3 ab=t.b-t.a, ac=t.c-t.a, ap=p-t.a;
    const float d1=dot(ab,ap), d2=dot(ac,ap);
    if(d1<=0&&d2<=0) return t.a;
    const Vec3 bp=p-t.b; const float d3=dot(ab,bp), d4=dot(ac,bp);
    if(d3>=0&&d4<=d3) return t.b;
    const float vc=d1*d4-d3*d2;
    if(vc<=0&&d1>=0&&d3<=0){const float v=d1/(d1-d3); return t.a+ab*v;}
    const Vec3 cp=p-t.c; const float d5=dot(ab,cp), d6=dot(ac,cp);
    if(d6>=0&&d5<=d6) return t.c;
    const float vb=d5*d2-d1*d6;
    if(vb<=0&&d2>=0&&d6<=0){const float w=d2/(d2-d6); return t.a+ac*w;}
    const float va=d3*d6-d5*d4;
    if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){const float w=(d4-d3)/((d4-d3)+(d5-d6)); return t.b+(t.c-t.b)*w;}
    const float inv=1.0f/(va+vb+vc); return t.a+ab*(vb*inv)+ac*(vc*inv);
}
inline bool segment_triangle(const Vec3& p0,const Vec3& p1,const Triangle& tri,float& out_t) {
    const Vec3 d=p1-p0, e1=tri.b-tri.a, e2=tri.c-tri.a;
    const Vec3 h=cross(d,e2); const float det=dot(e1,h);
    if(std::abs(det)<1.0e-10f) return false;
    const float inv=1.0f/det; const Vec3 s=p0-tri.a;
    const float u=inv*dot(s,h); if(u<0||u>1) return false;
    const Vec3 q=cross(s,e1); const float v=inv*dot(d,q); if(v<0||u+v>1) return false;
    const float t=inv*dot(e2,q); if(t<=1.0e-6f||t>=1.0f-1.0e-6f) return false;
    out_t=t; return true;
}

class TriangleBvh {
    struct Node { Aabb box; int left=-1,right=-1,start=0,count=0; bool leaf() const{return left<0;} };
    std::vector<Triangle> triangles_;
    std::vector<int> order_;
    std::vector<Node> nodes_;

    int build_node(int begin,int end) {
        const int index=static_cast<int>(nodes_.size()); nodes_.push_back({});
        Aabb box, centroids;
        for(int i=begin;i<end;++i){const Triangle& t=triangles_[order_[i]];box.grow(t.box);centroids.grow(t.centroid);}
        Node& n=nodes_[index]; n.box=box;
        const int count=end-begin;
        if(count<=8){n.start=begin;n.count=count;return index;}
        const Vec3 ext=centroids.hi-centroids.lo;
        const int axis=(ext.y>ext.x&&ext.y>=ext.z)?1:(ext.z>ext.x&&ext.z>ext.y?2:0);
        const int mid=begin+count/2;
        std::nth_element(order_.begin()+begin,order_.begin()+mid,order_.begin()+end,[&](int a,int b){
            const Vec3& ca=triangles_[a].centroid; const Vec3& cb=triangles_[b].centroid;
            return axis==0?ca.x<cb.x:(axis==1?ca.y<cb.y:ca.z<cb.z);
        });
        const int left=build_node(begin,mid),right=build_node(mid,end);
        nodes_[index].left=left;nodes_[index].right=right;return index;
    }
public:
    TriangleBvh()=default;
    TriangleBvh(const std::vector<Vec3>& vertices,const std::vector<int>& indices){reset(vertices,indices);}
    void reset(const std::vector<Vec3>& vertices,const std::vector<int>& indices) {
        if(indices.size()%3!=0) throw std::runtime_error("triangle index count must be divisible by three");
        for(const int index:indices)if(index<0||index>=static_cast<int>(vertices.size()))throw std::runtime_error("triangle index is outside the vertex array");
        std::vector<Triangle> candidates(indices.size()/3);std::vector<unsigned char> valid(candidates.size(),0);
        #pragma omp parallel for schedule(static)
        for(std::int64_t i=0;i<static_cast<std::int64_t>(candidates.size());++i){
            const int ia=indices[3*i],ib=indices[3*i+1],ic=indices[3*i+2];
            Triangle t; t.a=vertices[ia];t.b=vertices[ib];t.c=vertices[ic];
            const Vec3 normal=cross(t.b-t.a,t.c-t.a);if(length_sq(normal)<=1.0e-20f)continue;
            t.n=normalize_or(normal,{0,0,1});t.centroid=(t.a+t.b+t.c)/3.0f;
            t.box.grow(t.a);t.box.grow(t.b);t.box.grow(t.c);candidates[i]=t;valid[i]=1;
        }
        triangles_.clear();triangles_.reserve(candidates.size());for(std::size_t i=0;i<candidates.size();++i)if(valid[i])triangles_.push_back(candidates[i]);
        order_.resize(triangles_.size());std::iota(order_.begin(),order_.end(),0);nodes_.clear();nodes_.reserve(triangles_.size()*2);
        if(!triangles_.empty()) build_node(0,static_cast<int>(triangles_.size()));
    }
    bool empty() const{return triangles_.empty();}
    int triangle_count() const{return static_cast<int>(triangles_.size());}
    Nearest nearest(const Vec3& p,float max_distance=std::numeric_limits<float>::infinity()) const {
        Nearest best;best.distance_sq=max_distance*max_distance;if(nodes_.empty())return best;
        std::array<int,128> stack{};int sp=0;stack[sp++]=0;
        while(sp){const int ni=stack[--sp];const Node& n=nodes_[ni];if(distance_sq(n.box,p)>best.distance_sq)continue;
            if(n.leaf()){for(int k=0;k<n.count;++k){const int ti=order_[n.start+k];const Triangle& t=triangles_[ti];const Vec3 q=closest_triangle(p,t);const float d=length_sq(p-q);if(d<best.distance_sq){best={true,d,q,t.n,ti};}}}
            else {const float dl=distance_sq(nodes_[n.left].box,p),dr=distance_sq(nodes_[n.right].box,p);const int near=dl<dr?n.left:n.right,far=dl<dr?n.right:n.left;if(sp+2>static_cast<int>(stack.size()))throw std::runtime_error("BVH traversal stack overflow");stack[sp++]=far;stack[sp++]=near;}
        } return best;
    }
    Hit segment_hit(const Vec3& a,const Vec3& b) const {
        Hit best;if(nodes_.empty())return best;std::array<int,128> stack{};int sp=0;stack[sp++]=0;
        while(sp){const int ni=stack[--sp];const Node& n=nodes_[ni];if(!segment_aabb(n.box,a,b,best.t))continue;
            if(n.leaf()){for(int k=0;k<n.count;++k){const int ti=order_[n.start+k];float t;if(segment_triangle(a,b,triangles_[ti],t)&&t<best.t){best={true,t,lerp(a,b,t),triangles_[ti].n,ti};}}}
            else {if(sp+2>static_cast<int>(stack.size()))throw std::runtime_error("BVH traversal stack overflow");stack[sp++]=n.left;stack[sp++]=n.right;}
        } return best;
    }
};

struct Parameters {
    Vec3 gravity{0,0,-9.81f};
    float damping=0.05f,internal_damping=0.05f,max_velocity=5.0f;
    int iterations=20;
    float stretch_stiffness=1.0e4f,bend_stiffness=1.0e-3f;
    float collision_margin=0.0008f,collision_search=0.020f,collision_max_correction=0.005f;
    float collision_response=1.0f,collision_velocity_damping=0.5f,collision_smoothing=0.5f;
    int collision_passes=1,post_collision_iterations=2;
    float max_move_per_substep=0.001f;int max_substeps=16;
    bool keep_length=true,adaptive_root_lock=true;
    int settle_iterations=12;float settle_relaxation=0.5f,groom_strength=0.15f,repair_strength=0.4f;
    float length_tolerance=0.0001f,angle_change_limit_deg=30.0f,fold_limit_deg=90.0f;
    float roughness_factor=1.25f,settle_min_improvement=1.0e-4f;
    int settle_stagnation=3,collision_smooth_passes=4;
    int openmp_threads=0;
};

struct HeadFrame { Vec3 base{},up{0,0,1},tip{};float ear_offset=0; };

struct FrameStats {
    int substeps=1,hits=0,repaired_strands=0,settled_strands=0,failed_strands=0;
    int length_bad_before=0,length_bad_rods_before=0,folded_before=0,shape_bad_before=0,rough_bad_before=0,collision_bad_before=0;
    int shape_bad_after=0,collision_bad_after=0,max_settle_iterations=0;
    int adaptive_lock_max_points=0,adaptive_lock_strands=0;
    float auto_move_mm=0,max_length_error_mm=0,max_final_length_error_mm=0,max_angle_deg=0;
};

struct ShapeQuality {
    bool ok=true;
    float length_error=0.0f,max_angle_deg=0.0f,max_angle_change_deg=0.0f,roughness=0.0f,roughness_limit=0.0f,score=0.0f;
    int length_bad_rods=0,worst_joint=-1;
};

struct CollisionQuality {
    bool ok=true;
    int point_hits=0,segment_hits=0;
    float max_depth=0.0f;
};

class Simulator {
    int n_full_=0,pps_=0,segs_=0,guide_decimation_=1,n_guides_=0,root_locked_=1;
    float particle_mass_=0.003f;
    std::vector<int> guides_;
    std::vector<int> full_to_guide_nearest_; // n_full * guide_k
    std::vector<float> full_to_guide_weights_;
    int guide_k_=1;

    std::vector<Vec3> full_rest_positions_;
    std::vector<float> full_rest_lengths_;
    std::vector<float> full_rest_angles_deg_;
    std::vector<float> full_rest_roughness_;

    std::vector<Vec3> pos_,vel_,predicted_,pre_collision_,target_start_,target_end_;
    std::vector<float> inv_mass_,segment_rest_;
    std::vector<Quat> orient_;
    std::vector<Vec3> darboux_rest_;
    std::vector<int> lock_counts_;
    std::vector<Vec3> previous_eval_guides_;

    std::optional<HeadFrame> previous_head_;
    Vec3 head_velocity_ema_{};
    int last_adaptive_lock_max_=0,last_adaptive_lock_strands_=0;

    TriangleBvh body_,clothes_;

    int point_index(int strand,int local) const{return strand*pps_+local;}
    int segment_index(int strand,int local) const{return strand*segs_+local;}

    static float angle_deg(const Vec3& a,const Vec3& b) {
        return std::acos(clampf(dot(normalize_or(a),normalize_or(b)),-1.0f,1.0f))*180.0f/kPi;
    }

    void compute_full_rest_data() {
        full_rest_lengths_.resize(static_cast<std::size_t>(n_full_)*segs_);
        full_rest_angles_deg_.resize(static_cast<std::size_t>(n_full_)*std::max(0,pps_-2));
        full_rest_roughness_.resize(n_full_);
        #pragma omp parallel for schedule(static)
        for(int s=0;s<n_full_;++s){
            std::vector<Vec3> tangent(segs_),bend(std::max(0,segs_-1));
            for(int e=0;e<segs_;++e){const Vec3 d=full_rest_positions_[point_index(s,e+1)]-full_rest_positions_[point_index(s,e)];full_rest_lengths_[segment_index(s,e)]=std::max(length(d),1.0e-7f);tangent[e]=normalize_or(d,{0,0,-1});}
            float jerk_sum=0;
            for(int j=0;j<segs_-1;++j){const float a=angle_deg(tangent[j],tangent[j+1]);full_rest_angles_deg_[s*(segs_-1)+j]=a;bend[j]=tangent[j+1]-tangent[j];}
            for(int j=0;j<segs_-2;++j)jerk_sum+=length(bend[j+1]-bend[j]);
            full_rest_roughness_[s]=jerk_sum/static_cast<float>(std::max(1,segs_-2));
        }
    }

    void compute_guides() {
        for(int s=0;s<n_full_;s+=guide_decimation_) guides_.push_back(s);
        if(guides_.empty()) guides_.push_back(0);
        n_guides_=static_cast<int>(guides_.size());guide_k_=std::min(4,n_guides_);
        if(n_guides_==n_full_){guide_k_=1;full_to_guide_nearest_.resize(n_full_);full_to_guide_weights_.assign(n_full_,1.0f);std::iota(full_to_guide_nearest_.begin(),full_to_guide_nearest_.end(),0);return;}
        full_to_guide_nearest_.resize(static_cast<std::size_t>(n_full_)*guide_k_);
        full_to_guide_weights_.resize(static_cast<std::size_t>(n_full_)*guide_k_);
        #pragma omp parallel for schedule(static)
        for(int s=0;s<n_full_;++s){
            const Vec3 root=full_rest_positions_[point_index(s,0)];
            std::vector<std::pair<float,int>> distances;distances.reserve(n_guides_);
            for(int g=0;g<n_guides_;++g)distances.emplace_back(length_sq(root-full_rest_positions_[point_index(guides_[g],0)]),g);
            std::partial_sort(distances.begin(),distances.begin()+guide_k_,distances.end());
            if(distances[0].first<=1.0e-14f){for(int k=0;k<guide_k_;++k){full_to_guide_nearest_[s*guide_k_+k]=distances[0].second;full_to_guide_weights_[s*guide_k_+k]=k==0?1.0f:0.0f;}}
            else {float sum=0;for(int k=0;k<guide_k_;++k)sum+=1.0f/std::max(distances[k].first,1.0e-12f);for(int k=0;k<guide_k_;++k){full_to_guide_nearest_[s*guide_k_+k]=distances[k].second;full_to_guide_weights_[s*guide_k_+k]=(1.0f/std::max(distances[k].first,1.0e-12f))/sum;}}
        }
        for(int g=0;g<n_guides_;++g){const int s=guides_[g];for(int k=0;k<guide_k_;++k){full_to_guide_nearest_[s*guide_k_+k]=g;full_to_guide_weights_[s*guide_k_+k]=k==0?1.0f:0.0f;}}
    }

    void init_rod_state(const std::vector<Vec3>& initial) {
        pos_.resize(static_cast<std::size_t>(n_guides_)*pps_);vel_.assign(pos_.size(),{});predicted_.resize(pos_.size());pre_collision_.resize(pos_.size());
        target_start_.resize(pos_.size());target_end_.resize(pos_.size());previous_eval_guides_.resize(pos_.size());
        segment_rest_.resize(static_cast<std::size_t>(n_guides_)*segs_);orient_.resize(segment_rest_.size());darboux_rest_.resize(static_cast<std::size_t>(n_guides_)*std::max(0,segs_-1));
        lock_counts_.assign(n_guides_,root_locked_);inv_mass_.resize(pos_.size());
        #pragma omp parallel for schedule(static)
        for(int g=0;g<n_guides_;++g){
            const int fs=guides_[g];Quat prev_q{};Vec3 prev_t{};
            for(int j=0;j<pps_;++j){const Vec3 p=initial[point_index(fs,j)];const int i=g*pps_+j;pos_[i]=predicted_[i]=target_start_[i]=target_end_[i]=previous_eval_guides_[i]=p;inv_mass_[i]=j<root_locked_?0.0f:1.0f/particle_mass_;}
            // Rod constitutive data always comes from frame 1, even when the
            // requested simulation interval starts on a later frame.
            for(int e=0;e<segs_;++e){const Vec3 d=full_rest_positions_[point_index(fs,e+1)]-full_rest_positions_[point_index(fs,e)];const float l=std::max(length(d),1.0e-7f);const Vec3 t=normalize_or(d,{0,0,1});segment_rest_[g*segs_+e]=l;Quat q=e==0?shortest_arc({0,0,1},t):normalized(shortest_arc(prev_t,t)*prev_q);orient_[g*segs_+e]=q;prev_q=q;prev_t=t;}
            for(int e=0;e<segs_-1;++e){const Quat q=relative_darboux(orient_[g*segs_+e],orient_[g*segs_+e+1]);darboux_rest_[g*(segs_-1)+e]={q.x,q.y,q.z};}
        }
    }

    void update_inverse_mass() {
        #pragma omp parallel for schedule(static)
        for(int g=0;g<n_guides_;++g)for(int j=0;j<pps_;++j)inv_mass_[g*pps_+j]=j<lock_counts_[g]?0.0f:1.0f/particle_mass_;
    }

    void position_solve(int g,float dt,float kss) {
        const int base=g*pps_,f0=lock_counts_[g];if(f0>=pps_)return;
        std::vector<double> c(pps_),dx(pps_),dy(pps_),dz(pps_);const double inv_dt2=1.0/std::max(static_cast<double>(dt)*dt,1.0e-12);
        for(int r=f0;r<pps_;++r){const int gi=base+r;const double mass=1.0/static_cast<double>(inv_mass_[gi]);double diag=mass*inv_dt2;double rx=pre_collision_[gi].x*(mass*inv_dt2),ry=pre_collision_[gi].y*(mass*inv_dt2),rz=pre_collision_[gi].z*(mass*inv_dt2);
            if(r<pps_-1){diag+=kss;const int e=g*segs_+r;const Vec3 term=rotate(orient_[e],{0,0,1})*(kss*segment_rest_[e]);rx-=term.x;ry-=term.y;rz-=term.z;}
            if(r>0){diag+=kss;const int e=g*segs_+r-1;const Vec3 term=rotate(orient_[e],{0,0,1})*(kss*segment_rest_[e]);rx+=term.x;ry+=term.y;rz+=term.z;}
            double sub=0.0,sup=r<pps_-1?-static_cast<double>(kss):0.0;if(r>f0)sub=-static_cast<double>(kss);else if(r>0){rx+=predicted_[gi-1].x*kss;ry+=predicted_[gi-1].y*kss;rz+=predicted_[gi-1].z*kss;}
            const double denom=r>f0?diag-sub*c[r-1]:diag;c[r]=sup/denom;if(r==f0){dx[r]=rx/denom;dy[r]=ry/denom;dz[r]=rz/denom;}else{dx[r]=(rx-dx[r-1]*sub)/denom;dy[r]=(ry-dy[r-1]*sub)/denom;dz[r]=(rz-dz[r-1]*sub)/denom;}
        }
        for(int r=pps_-2;r>=f0;--r){dx[r]-=dx[r+1]*c[r];dy[r]-=dy[r+1]*c[r];dz[r]-=dz[r+1]*c[r];}for(int r=f0;r<pps_;++r)predicted_[base+r]={static_cast<float>(dx[r]),static_cast<float>(dy[r]),static_cast<float>(dz[r])};
    }

    void orientation_solve(int g,float kss,float kbt) {
        const Mat3 I=Mat3::identity();
        for(int parity=0;parity<2;++parity)for(int local=parity;local<segs_;local+=2){const int e=g*segs_+local,i=g*pps_+local,j=i+1;Quat q=orient_[e];const Vec3 d3=rotate(q,{0,0,1});const float l=segment_rest_[e];const Vec3 residual=(predicted_[j]-predicted_[i])-d3*l;
            const Vec3 col0=-l*rotate(q,{0,-1,0}),col1=-l*rotate(q,{1,0,0});Mat3 A;A.m[0][0]=kss*dot(col0,col0);A.m[0][1]=A.m[1][0]=kss*dot(col0,col1);A.m[1][1]=kss*dot(col1,col1);Vec3 rhs{-kss*dot(col0,residual),-kss*dot(col1,residual),0.0f};
            const float k_eff=kbt*(1.0f+4.0f*(1.0f-static_cast<float>(local)/std::max(1,segs_)));
            auto add_bend=[&](const Quat& rel,const Vec3& rest,bool right){const Vec3 v{rel.x,rel.y,rel.z};const Vec3 br=v-rest;Mat3 J;const float w=rel.w;
                if(right){J.m[0][0]=-0.5f*w;J.m[0][1]=-0.5f*v.z;J.m[0][2]=0.5f*v.y;J.m[1][0]=0.5f*v.z;J.m[1][1]=-0.5f*w;J.m[1][2]=-0.5f*v.x;J.m[2][0]=-0.5f*v.y;J.m[2][1]=0.5f*v.x;J.m[2][2]=-0.5f*w;}
                else {J.m[0][0]=0.5f*w;J.m[0][1]=-0.5f*v.z;J.m[0][2]=0.5f*v.y;J.m[1][0]=0.5f*v.z;J.m[1][1]=0.5f*w;J.m[1][2]=-0.5f*v.x;J.m[2][0]=-0.5f*v.y;J.m[2][1]=0.5f*v.x;J.m[2][2]=0.5f*w;}
                Vec3 cols[3]{{J.m[0][0],J.m[1][0],J.m[2][0]},{J.m[0][1],J.m[1][1],J.m[2][1]},{J.m[0][2],J.m[1][2],J.m[2][2]}};
                for(int a=0;a<3;++a){component(rhs,a)-=k_eff*dot(cols[a],br);for(int b=0;b<3;++b)A.m[a][b]+=k_eff*dot(cols[a],cols[b]);}
            };
            if(local<segs_-1)add_bend(relative_darboux(q,orient_[e+1]),darboux_rest_[g*(segs_-1)+local],true);
            if(local>0)add_bend(relative_darboux(orient_[e-1],q),darboux_rest_[g*(segs_-1)+local-1],false);
            A=A+I*1.0e-7f;if(std::abs(determinant(A))<1.0e-20f)continue;Vec3 omega=inverse(A)*rhs;const float mag=length(omega);if(mag>0.5f)omega*=0.5f/mag;orient_[e]=normalized(q*exp_quat(omega));
        }
    }

    void solve_strand(int g,float dt,int iterations,float kss,float kbt) {
        const int base=g*pps_;for(int j=0;j<pps_;++j)pre_collision_[base+j]=predicted_[base+j];
        for(int it=0;it<std::max(1,iterations);++it){position_solve(g,dt,kss);orientation_solve(g,kss,kbt);}
    }

    static Vec3 clamp_correction(Vec3 c,float limit) {const float n=length(c);return n>limit&&n>kEps?c*(limit/n):c;}

    bool point_collision(const TriangleBvh& mesh,bool body,int g,int j,const Parameters& p,bool allow_sweep,int& hits) {
        const int i=g*pps_+j;if(inv_mass_[i]<=0)return false;const Vec3 old=pos_[i];Vec3 next=predicted_[i];Vec3 normal{};bool contact=false;
        if(allow_sweep){Hit h=mesh.segment_hit(old,next);const Vec3 delta=next-old;if(h.found&&(!body||dot(delta,h.n)<0)){normal=h.n;if(!body&&dot(delta,normal)>0)normal=-normal;const Vec3 target=h.p+normal*p.collision_margin;next+=clamp_correction(target-next,p.collision_max_correction)*p.collision_response;contact=true;}}
        if(!contact){Nearest q=mesh.nearest(next,body?std::numeric_limits<float>::infinity():p.collision_search);if(q.found){const Vec3 sd=next-q.p;const float ud=std::sqrt(q.distance_sq);if(body){normal=q.n;const float signed_distance=dot(sd,normal);if(signed_distance<p.collision_margin){next+=clamp_correction(normal*(p.collision_margin-signed_distance),p.collision_max_correction)*p.collision_response;contact=true;}}
                else if(ud<p.collision_margin){normal=ud>kEps?sd/ud:q.n;const Vec3 target=q.p+normal*p.collision_margin;next+=clamp_correction(target-next,p.collision_max_correction)*p.collision_response;contact=true;}}}
        if(contact){const float vn=dot(vel_[i],normal);if(vn<0)vel_[i]-=normal*vn;predicted_[i]=next;++hits;}return contact;
    }

    bool segment_collisions(const TriangleBvh& mesh,int g,const Parameters& p,int& hits) {
        bool contact=false;for(int pass=0;pass<std::max(1,p.collision_passes);++pass)for(int parity=0;parity<2;++parity)for(int e=parity;e<segs_;e+=2){const int i=g*pps_+e,j=i+1;if(inv_mass_[j]<=0)continue;Hit h=mesh.segment_hit(predicted_[i],predicted_[j]);if(!h.found)continue;Vec3 n=h.n;const Vec3 d=predicted_[j]-predicted_[i];if(dot(d,n)>0)n=-n;const Vec3 target=h.p+n*p.collision_margin;predicted_[j]+=clamp_correction(target-predicted_[j],p.collision_max_correction)*p.collision_response;const float vn=dot(vel_[j],n);if(vn<0)vel_[j]-=n*vn;contact=true;++hits;}
        return contact;
    }

    bool collide_strand(int g,const Parameters& p,bool allow_sweep,int& hits) {
        const int base=g*pps_;std::vector<Vec3> before(pps_),correction(pps_),tmp(pps_);for(int j=0;j<pps_;++j)before[j]=predicted_[base+j];bool contact=false;
        for(int j=0;j<pps_;++j){if(!body_.empty())contact|=point_collision(body_,true,g,j,p,allow_sweep,hits);if(!clothes_.empty())contact|=point_collision(clothes_,false,g,j,p,allow_sweep,hits);}
        if(!body_.empty())contact|=segment_collisions(body_,g,p,hits);if(!clothes_.empty())contact|=segment_collisions(clothes_,g,p,hits);
        const int passes=static_cast<int>(std::lround(clampf(p.collision_smoothing,0,1)*8.0f));if(passes>0&&contact){for(int j=0;j<pps_;++j)correction[j]=inv_mass_[base+j]>0?predicted_[base+j]-before[j]:Vec3{};for(int pass=0;pass<passes;++pass){for(int j=0;j<pps_;++j){if(inv_mass_[base+j]<=0){tmp[j]={};continue;}Vec3 sum{};int count=0;if(j>0){sum+=correction[j-1];++count;}if(j+1<pps_){sum+=correction[j+1];++count;}tmp[j]=count?lerp(correction[j],sum/static_cast<float>(count),0.5f):correction[j];}correction.swap(tmp);}for(int j=0;j<pps_;++j)if(inv_mass_[base+j]>0)predicted_[base+j]=before[j]+correction[j];}
        return contact;
    }

    ShapeQuality evaluate_shape(const std::vector<Vec3>& points,int s,const Parameters& p) const {
        ShapeQuality q;const int base=s*pps_;std::vector<Vec3> tangent(segs_),bend(std::max(0,segs_-1));float jerk_sum=0;
        for(int e=0;e<segs_;++e){const Vec3 d=points[base+e+1]-points[base+e];const float l=length(d);const float error=std::abs(l-full_rest_lengths_[s*segs_+e]);q.length_error=std::max(q.length_error,error);if(error>p.length_tolerance)++q.length_bad_rods;tangent[e]=normalize_or(d,{0,0,-1});}
        for(int j=0;j<segs_-1;++j){const float a=angle_deg(tangent[j],tangent[j+1]);const float delta=std::abs(a-full_rest_angles_deg_[s*(segs_-1)+j]);if(a>q.max_angle_deg){q.max_angle_deg=a;q.worst_joint=j+1;}q.max_angle_change_deg=std::max(q.max_angle_change_deg,delta);bend[j]=tangent[j+1]-tangent[j];}
        for(int j=0;j<segs_-2;++j)jerk_sum+=length(bend[j+1]-bend[j]);
        q.roughness=jerk_sum/static_cast<float>(std::max(1,segs_-2));
        // Mean absolute second difference of unit tangents (radians-like).
        // The absolute floor accepts gentle curvature redistribution while a
        // sharp or alternating kink remains well above it.
        q.roughness_limit=std::max(0.15f,full_rest_roughness_[s]*p.roughness_factor+0.02f);
        const bool length_ok=!p.keep_length||q.length_error<=p.length_tolerance;
        q.ok=length_ok&&q.max_angle_deg<=p.fold_limit_deg&&q.max_angle_change_deg<=p.angle_change_limit_deg&&q.roughness<=q.roughness_limit;
        q.score=(p.keep_length?q.length_error/std::max(p.length_tolerance,1.0e-8f):0.0f)+std::max(0.0f,q.max_angle_deg-p.fold_limit_deg)/10.0f+std::max(0.0f,q.max_angle_change_deg-p.angle_change_limit_deg)/10.0f+std::max(0.0f,q.roughness-q.roughness_limit)*10.0f;
        return q;
    }

    CollisionQuality evaluate_collision(const std::vector<Vec3>& points,int s,const Parameters& p) const {
        CollisionQuality q;const int base=s*pps_;const float accepted_margin=std::max(0.0f,p.collision_margin-std::max(1.0e-7f,p.collision_margin*1.0e-4f));
        for(int j=std::max(1,root_locked_);j<pps_;++j){const Vec3 x=points[base+j];if(!body_.empty()){Nearest n=body_.nearest(x);if(n.found){const float signed_d=dot(x-n.p,n.n);if(signed_d<accepted_margin){++q.point_hits;q.max_depth=std::max(q.max_depth,p.collision_margin-signed_d);}}}if(!clothes_.empty()){Nearest n=clothes_.nearest(x,p.collision_search);if(n.found){const float d=std::sqrt(n.distance_sq);if(d<accepted_margin){++q.point_hits;q.max_depth=std::max(q.max_depth,p.collision_margin-d);}}}}
        for(int e=std::max(0,root_locked_-1);e<segs_;++e){if(!body_.empty()&&body_.segment_hit(points[base+e],points[base+e+1]).found)++q.segment_hits;if(!clothes_.empty()&&clothes_.segment_hit(points[base+e],points[base+e+1]).found)++q.segment_hits;}
        q.ok=q.point_hits==0&&q.segment_hits==0;return q;
    }

    void project_lengths(std::vector<Vec3>& points,int s,int locked) const {
        const int base=s*pps_;locked=std::max(1,std::min(locked,pps_));
        for(int j=locked;j<pps_;++j){const float l=full_rest_lengths_[s*segs_+j-1];const Vec3 fallback=full_rest_positions_[base+j]-full_rest_positions_[base+j-1];const Vec3 direction=normalize_or(points[base+j]-points[base+j-1],fallback);points[base+j]=points[base+j-1]+direction*l;}
    }

    void smooth_shape(std::vector<Vec3>& points,int s,int locked,float strength,bool keep_length,const ShapeQuality* quality=nullptr) const {
        const int base=s*pps_;std::vector<Vec3> t(segs_),out(segs_);std::vector<float> lengths(segs_);for(int e=0;e<segs_;++e){const Vec3 edge=points[base+e+1]-points[base+e];lengths[e]=keep_length?full_rest_lengths_[s*segs_+e]:std::max(length(edge),1.0e-7f);t[e]=normalize_or(edge,full_rest_positions_[base+e+1]-full_rest_positions_[base+e]);}out=t;
        const int first=std::max(0,locked-1);for(int e=first;e<segs_;++e){float local_strength=strength;if(quality&&quality->worst_joint>=0&&std::abs(e-quality->worst_joint)>2)local_strength*=0.2f;Vec3 avg{};float w=0;if(e>0){avg+=t[e-1];w+=1;}avg+=t[e]*2.0f;w+=2;if(e+1<segs_){avg+=t[e+1];w+=1;}Vec3 target=normalize_or(avg/w,t[e]);if(quality&&quality->max_angle_deg>90.0f&&quality->worst_joint==e&&e>0){target=t[e-1];local_strength=1.0f;}out[e]=slerp_direction(t[e],target,local_strength);}
        for(int j=locked;j<pps_;++j)points[base+j]=points[base+j-1]+out[j-1]*lengths[j-1];
    }

    void relax_strand(std::vector<Vec3>& points,int s,int locked,const std::vector<Vec3>& before,float relaxation) const {
        const int base=s*pps_;const float alpha=clampf(relaxation,0.01f,1.0f);
        for(int j=locked;j<pps_;++j)points[base+j]=lerp(before[j],points[base+j],alpha);
    }

    bool repair_collision(std::vector<Vec3>& points,int s,const Parameters& p) const {
        const int base=s*pps_,locked=root_locked_;std::vector<Vec3> before(pps_),corr(pps_),tmp(pps_);for(int j=0;j<pps_;++j)before[j]=points[base+j];bool changed=false;
        for(int j=locked;j<pps_;++j){Vec3& x=points[base+j];if(!body_.empty()){Nearest n=body_.nearest(x);if(n.found){const float signed_d=dot(x-n.p,n.n);if(signed_d<p.collision_margin){x+=clamp_correction(n.n*(p.collision_margin-signed_d),p.collision_max_correction);changed=true;}}}if(!clothes_.empty()){Nearest n=clothes_.nearest(x,p.collision_search);if(n.found&&n.distance_sq<p.collision_margin*p.collision_margin){const float d=std::sqrt(n.distance_sq);const Vec3 dir=d>kEps?(x-n.p)/d:n.n;x+=clamp_correction(n.p+dir*p.collision_margin-x,p.collision_max_correction);changed=true;}}}
        for(int e=std::max(0,locked-1);e<segs_;++e){const int j=e+1;if(!body_.empty()){Hit h=body_.segment_hit(points[base+e],points[base+j]);if(h.found){Vec3 n=h.n;if(dot(points[base+j]-points[base+e],n)>0)n=-n;points[base+j]+=clamp_correction(h.p+n*p.collision_margin-points[base+j],p.collision_max_correction);changed=true;}}if(!clothes_.empty()){Hit h=clothes_.segment_hit(points[base+e],points[base+j]);if(h.found){Vec3 n=h.n;if(dot(points[base+j]-points[base+e],n)>0)n=-n;points[base+j]+=clamp_correction(h.p+n*p.collision_margin-points[base+j],p.collision_max_correction);changed=true;}}}
        if(changed&&p.collision_smooth_passes>0){for(int j=0;j<pps_;++j)corr[j]=j<locked?Vec3{}:points[base+j]-before[j];for(int pass=0;pass<p.collision_smooth_passes;++pass){for(int j=0;j<pps_;++j){if(j<locked){tmp[j]={};continue;}Vec3 sum{};int count=0;if(j>0){sum+=corr[j-1];++count;}if(j+1<pps_){sum+=corr[j+1];++count;}tmp[j]=count?lerp(corr[j],sum/static_cast<float>(count),0.5f):corr[j];}corr.swap(tmp);}for(int j=locked;j<pps_;++j)points[base+j]=before[j]+corr[j];}
        return changed;
    }

    struct SettleCounters {int repaired=0,settled=0,failed=0,length_bad=0,length_bad_rods=0,folded=0,shape_bad=0,rough_bad=0,collision_bad=0,shape_bad_after=0,collision_bad_after=0,max_iterations=0;float max_length_error=0,max_final_length_error=0,max_angle=0;};

    SettleCounters settle_full(std::vector<Vec3>& points,const Parameters& p,std::vector<int>& repaired_mask) const {
        repaired_mask.assign(n_full_,0);SettleCounters total;
        #pragma omp parallel
        {
            SettleCounters local;
            #pragma omp for schedule(dynamic,16)
            for(int s=0;s<n_full_;++s){const int base=s*pps_;std::vector<Vec3> original(points.begin()+base,points.begin()+base+pps_);ShapeQuality initial=evaluate_shape(points,s,p);CollisionQuality initial_collision=evaluate_collision(points,s,p);if(initial.length_bad_rods>0)++local.length_bad;local.length_bad_rods+=initial.length_bad_rods;if(initial.max_angle_deg>p.fold_limit_deg)++local.folded;if(!initial.ok)++local.shape_bad;if(initial.roughness>initial.roughness_limit)++local.rough_bad;if(!initial_collision.ok)++local.collision_bad;local.max_length_error=std::max(local.max_length_error,initial.length_error);local.max_angle=std::max(local.max_angle,initial.max_angle_deg);
                if(initial.ok&&initial_collision.ok){++local.settled;local.max_final_length_error=std::max(local.max_final_length_error,initial.length_error);continue;}
                repaired_mask[s]=1;++local.repaired;std::vector<Vec3> best=original;float best_score=std::numeric_limits<float>::infinity(),previous=best_score;int stagnation=0,used=0;bool converged=false;
                std::vector<Vec3> before=original;if(!initial.ok)smooth_shape(points,s,root_locked_,p.groom_strength,p.keep_length,&initial);else repair_collision(points,s,p);relax_strand(points,s,root_locked_,before,p.settle_relaxation);if(p.keep_length)project_lengths(points,s,root_locked_);
                for(int iter=0;iter<p.settle_iterations;++iter){used=iter+1;ShapeQuality shape=evaluate_shape(points,s,p);CollisionQuality collision=evaluate_collision(points,s,p);const float score=shape.score+collision.point_hits*10.0f+collision.segment_hits*20.0f+collision.max_depth/std::max(p.collision_margin,1.0e-8f);
                    if(score<best_score){best_score=score;std::copy(points.begin()+base,points.begin()+base+pps_,best.begin());}
                    if(shape.ok&&collision.ok){converged=true;break;}
                    if(previous-score<p.settle_min_improvement)++stagnation;else stagnation=0;previous=score;
                    // Repairing a 180-degree reversal can move the single worst
                    // fold one rod toward the tip without lowering the maximum
                    // angle yet.  Do not mistake that finite propagation for a
                    // stagnant loop; the hard iteration cap still applies.
                    if(stagnation>=p.settle_stagnation&&shape.max_angle_deg<=p.fold_limit_deg&&collision.ok)break;
                    before.assign(points.begin()+base,points.begin()+base+pps_);
                    if(!shape.ok)smooth_shape(points,s,root_locked_,p.repair_strength,p.keep_length,&shape);else repair_collision(points,s,p);
                    relax_strand(points,s,root_locked_,before,p.settle_relaxation);if(p.keep_length)project_lengths(points,s,root_locked_);
                }
                local.max_iterations=std::max(local.max_iterations,used);if(converged)++local.settled;else {++local.failed;std::copy(best.begin(),best.end(),points.begin()+base);}
                const ShapeQuality final_shape=evaluate_shape(points,s,p);const CollisionQuality final_collision=evaluate_collision(points,s,p);if(!final_shape.ok)++local.shape_bad_after;if(!final_collision.ok)++local.collision_bad_after;local.max_final_length_error=std::max(local.max_final_length_error,final_shape.length_error);
            }
            #pragma omp critical
            {total.repaired+=local.repaired;total.settled+=local.settled;total.failed+=local.failed;total.length_bad+=local.length_bad;total.length_bad_rods+=local.length_bad_rods;total.folded+=local.folded;total.shape_bad+=local.shape_bad;total.rough_bad+=local.rough_bad;total.collision_bad+=local.collision_bad;total.shape_bad_after+=local.shape_bad_after;total.collision_bad_after+=local.collision_bad_after;total.max_iterations=std::max(total.max_iterations,local.max_iterations);total.max_length_error=std::max(total.max_length_error,local.max_length_error);total.max_final_length_error=std::max(total.max_final_length_error,local.max_final_length_error);total.max_angle=std::max(total.max_angle,local.max_angle);}
        }
        return total;
    }

    std::vector<Vec3> gather_guides(const std::vector<Vec3>& full) const {
        std::vector<Vec3> out(static_cast<std::size_t>(n_guides_)*pps_);
        #pragma omp parallel for schedule(static)
        for(int g=0;g<n_guides_;++g)for(int j=0;j<pps_;++j)out[g*pps_+j]=full[point_index(guides_[g],j)];return out;
    }

    std::vector<Vec3> restore_full(const std::vector<Vec3>& evaluated,const std::vector<Vec3>& evaluated_guides) const {
        if(n_guides_==n_full_)return pos_;std::vector<Vec3> out(evaluated.size());
        #pragma omp parallel for schedule(static)
        for(int s=0;s<n_full_;++s)for(int j=0;j<pps_;++j){Vec3 delta{};for(int k=0;k<guide_k_;++k){const int g=full_to_guide_nearest_[s*guide_k_+k];delta+=(pos_[g*pps_+j]-evaluated_guides[g*pps_+j])*full_to_guide_weights_[s*guide_k_+k];}out[point_index(s,j)]=evaluated[point_index(s,j)]+delta;}
        return out;
    }

    void adaptive_locks(const std::vector<Vec3>& current_guides,float dt,const std::optional<HeadFrame>& head,const Parameters& p) {
        std::fill(lock_counts_.begin(),lock_counts_.end(),root_locked_);last_adaptive_lock_max_=root_locked_;last_adaptive_lock_strands_=0;if(!p.adaptive_root_lock||!head){previous_head_=head;update_inverse_mass();return;}
        if(previous_head_){const Vec3 raw=(head->tip-previous_head_->tip)/std::max(dt,1.0e-6f);head_velocity_ema_=head_velocity_ema_*0.5f+raw*0.5f;}previous_head_=head;const float speed=length(head_velocity_ema_);if(speed<0.003f){update_inverse_mass();return;}const Vec3 v=head_velocity_ema_/speed;const float cos_inner=std::cos(75.0f*kPi/180.0f),cos_outer=std::cos(90.0f*kPi/180.0f);
        #pragma omp parallel for schedule(static)
        for(int g=0;g<n_guides_;++g){Vec3 rd=normalize_or(current_guides[g*pps_]-head->base,{0,0,1});const float c=dot(rd,v);float w=clampf((c-cos_outer)/(cos_inner-cos_outer),0,1);w=w*w*(3-2*w);int prefix=0;for(int j=0;j<pps_;++j){const float h=dot(current_guides[g*pps_+j]-head->base,head->up);if(h>=head->ear_offset)++prefix;else break;}const int target=std::max(root_locked_,prefix);lock_counts_[g]=std::max(1,std::min(pps_,static_cast<int>(std::lround(root_locked_+(target-root_locked_)*w))));}
        for(const int count:lock_counts_){last_adaptive_lock_max_=std::max(last_adaptive_lock_max_,count);if(count>root_locked_)++last_adaptive_lock_strands_;}update_inverse_mass();
    }

    float estimate_move_mm(const std::vector<Vec3>& target,float dt,const Parameters& p) const {
        float max_move=0;
        #pragma omp parallel
        {
            float local_max=0;
            #pragma omp for schedule(static)
            for(int g=0;g<n_guides_;++g)for(int j=0;j<pps_;++j){float m=0;if(j<lock_counts_[g])m=length(target[g*pps_+j]-previous_eval_guides_[g*pps_+j]);else m=length(vel_[g*pps_+j]*dt+p.gravity*(0.5f*dt*dt));local_max=std::max(local_max,m);}
            #pragma omp critical
            {max_move=std::max(max_move,local_max);}
        }
        return max_move*1000.0f;
    }

    void simulate_substeps(const std::vector<Vec3>& next_targets,int substeps,float dt_frame,const Parameters& p,FrameStats& stats) {
        target_start_=previous_eval_guides_;target_end_=next_targets;const float dt=dt_frame/std::max(1,substeps);
        for(int step=0;step<substeps;++step){const float alpha=static_cast<float>(step+1)/substeps;
            #pragma omp parallel for schedule(static)
            for(int g=0;g<n_guides_;++g){const int base=g*pps_;for(int j=0;j<pps_;++j){const int i=base+j;if(inv_mass_[i]<=0){const Vec3 target=lerp(target_start_[i],target_end_[i],alpha);pos_[i]=predicted_[i]=target;vel_[i]={};}else{vel_[i]+=p.gravity*dt;const float speed=length(vel_[i]);if(p.max_velocity>0&&speed>p.max_velocity)vel_[i]*=p.max_velocity/speed;predicted_[i]=pos_[i]+vel_[i]*dt;}}
                solve_strand(g,dt,p.iterations,p.stretch_stiffness,p.bend_stiffness);std::vector<Vec3> velocity_source(pps_);for(int j=0;j<pps_;++j)velocity_source[j]=predicted_[base+j];int local_hits=0;bool contact=collide_strand(g,p,true,local_hits);
                for(int k=0;k<std::max(0,p.post_collision_iterations);++k){solve_strand(g,dt,1,p.stretch_stiffness,p.bend_stiffness);contact|=collide_strand(g,p,false,local_hits);}solve_strand(g,dt,std::max(2,p.post_collision_iterations),p.stretch_stiffness,p.bend_stiffness);
                std::vector<Vec3> new_vel(pps_);for(int j=0;j<pps_;++j){const int i=base+j;if(inv_mass_[i]<=0)new_vel[j]={};else{Vec3 v=(velocity_source[j]-pos_[i])/dt*(1.0f-p.damping);const float speed=length(v);if(p.max_velocity>0&&speed>p.max_velocity)v*=p.max_velocity/speed;if(contact)v*=1.0f-p.collision_velocity_damping;new_vel[j]=v;}}
                if(p.internal_damping>0){const float mu=clampf(p.internal_damping,0,0.5f);const std::vector<Vec3> snapshot=new_vel;for(int j=0;j<pps_;++j){const int i=base+j;if(inv_mass_[i]<=0)continue;Vec3 lap{};if(j>0&&inv_mass_[i-1]>0)lap+=snapshot[j-1]-snapshot[j];if(j+1<pps_&&inv_mass_[i+1]>0)lap+=snapshot[j+1]-snapshot[j];new_vel[j]+=lap*mu;}}
                for(int j=0;j<pps_;++j){pos_[base+j]=predicted_[base+j];vel_[base+j]=new_vel[j];}
                #pragma omp atomic
                stats.hits+=local_hits;
            }
        }
    }

public:
    Simulator(const std::vector<Vec3>& initial,const std::vector<Vec3>& frame1_rest,int pps,int root_locked,float particle_mass,int guide_decimation)
        :pps_(pps),segs_(pps-1),guide_decimation_(std::max(1,guide_decimation)),root_locked_(std::max(1,std::min(root_locked,pps))),particle_mass_(std::max(particle_mass,1.0e-8f)),full_rest_positions_(frame1_rest) {
        if(!std::isfinite(particle_mass)||pps_<3||initial.empty()||initial.size()!=frame1_rest.size()||initial.size()%pps_!=0)throw std::runtime_error("invalid simulator inputs or position topology");n_full_=static_cast<int>(initial.size()/pps_);compute_full_rest_data();compute_guides();init_rod_state(initial);
    }

    int strand_count() const{return n_full_;}int points_per_strand() const{return pps_;}int guide_count() const{return n_guides_;}
    void prime_head(const std::optional<HeadFrame>& h){previous_head_=h;head_velocity_ema_={};}

    FrameStats audit(const std::vector<Vec3>& evaluated,const std::vector<Vec3>& body_vertices,const std::vector<int>& body_indices,const std::vector<Vec3>& clothes_vertices,const std::vector<int>& clothes_indices,const Parameters& p) {
        if(evaluated.size()!=full_rest_positions_.size())throw std::runtime_error("audit target topology changed");body_.reset(body_vertices,body_indices);clothes_.reset(clothes_vertices,clothes_indices);FrameStats total;
        #pragma omp parallel
        {
            FrameStats local;
            #pragma omp for schedule(dynamic,16)
            for(int s=0;s<n_full_;++s){const ShapeQuality shape=evaluate_shape(evaluated,s,p);const CollisionQuality collision=evaluate_collision(evaluated,s,p);if(shape.length_bad_rods>0)++local.length_bad_before;local.length_bad_rods_before+=shape.length_bad_rods;if(shape.max_angle_deg>p.fold_limit_deg)++local.folded_before;if(!shape.ok)++local.shape_bad_before;if(shape.roughness>shape.roughness_limit)++local.rough_bad_before;if(!collision.ok)++local.collision_bad_before;if(shape.ok&&collision.ok)++local.settled_strands;else ++local.repaired_strands;local.shape_bad_after=local.shape_bad_before;local.collision_bad_after=local.collision_bad_before;local.max_length_error_mm=std::max(local.max_length_error_mm,shape.length_error*1000.0f);local.max_final_length_error_mm=local.max_length_error_mm;local.max_angle_deg=std::max(local.max_angle_deg,shape.max_angle_deg);}
            #pragma omp critical
            {total.repaired_strands+=local.repaired_strands;total.settled_strands+=local.settled_strands;total.length_bad_before+=local.length_bad_before;total.length_bad_rods_before+=local.length_bad_rods_before;total.folded_before+=local.folded_before;total.shape_bad_before+=local.shape_bad_before;total.rough_bad_before+=local.rough_bad_before;total.collision_bad_before+=local.collision_bad_before;total.shape_bad_after+=local.shape_bad_after;total.collision_bad_after+=local.collision_bad_after;total.max_length_error_mm=std::max(total.max_length_error_mm,local.max_length_error_mm);total.max_final_length_error_mm=std::max(total.max_final_length_error_mm,local.max_final_length_error_mm);total.max_angle_deg=std::max(total.max_angle_deg,local.max_angle_deg);}
        }
        return total;
    }

    std::pair<std::vector<Vec3>,FrameStats> frame(const std::vector<Vec3>& evaluated,const std::vector<Vec3>& body_vertices,const std::vector<int>& body_indices,const std::vector<Vec3>& clothes_vertices,const std::vector<int>& clothes_indices,float dt_frame,const Parameters& p,const std::optional<HeadFrame>& head,std::vector<int>& repaired_mask) {
        if(evaluated.size()!=full_rest_positions_.size())throw std::runtime_error("frame target topology changed");if(!std::isfinite(dt_frame)||dt_frame<=0.0f)throw std::runtime_error("dt_frame must be finite and positive");body_.reset(body_vertices,body_indices);clothes_.reset(clothes_vertices,clothes_indices);const std::vector<Vec3> eval_guides=gather_guides(evaluated);adaptive_locks(eval_guides,dt_frame,head,p);FrameStats stats;stats.auto_move_mm=estimate_move_mm(eval_guides,dt_frame,p);stats.substeps=std::min(std::max(1,p.max_substeps),std::max(1,static_cast<int>(std::ceil(stats.auto_move_mm*0.001f/std::max(p.max_move_per_substep,1.0e-6f)))));simulate_substeps(eval_guides,stats.substeps,dt_frame,p,stats);std::vector<Vec3> full=restore_full(evaluated,eval_guides);
        SettleCounters settle=settle_full(full,p,repaired_mask);stats.repaired_strands=settle.repaired;stats.settled_strands=settle.settled;stats.failed_strands=settle.failed;stats.length_bad_before=settle.length_bad;stats.length_bad_rods_before=settle.length_bad_rods;stats.folded_before=settle.folded;stats.shape_bad_before=settle.shape_bad;stats.rough_bad_before=settle.rough_bad;stats.collision_bad_before=settle.collision_bad;stats.shape_bad_after=settle.shape_bad_after;stats.collision_bad_after=settle.collision_bad_after;stats.max_settle_iterations=settle.max_iterations;stats.adaptive_lock_max_points=last_adaptive_lock_max_;stats.adaptive_lock_strands=last_adaptive_lock_strands_;stats.max_length_error_mm=settle.max_length_error*1000.0f;stats.max_final_length_error_mm=settle.max_final_length_error*1000.0f;stats.max_angle_deg=settle.max_angle;
        for(int g=0;g<n_guides_;++g){const int fs=guides_[g];bool repaired=repaired_mask[fs]!=0;for(int j=0;j<pps_;++j){pos_[g*pps_+j]=predicted_[g*pps_+j]=full[point_index(fs,j)];if(repaired)vel_[g*pps_+j]={};previous_eval_guides_[g*pps_+j]=pos_[g*pps_+j];}}
        return {std::move(full),stats};
    }

    std::pair<std::vector<Vec3>,FrameStats> settle_external(const std::vector<Vec3>& evaluated,const Parameters& p,std::vector<int>& repaired_mask) {
        if(evaluated.size()!=full_rest_positions_.size())throw std::runtime_error("external settle topology changed");std::vector<Vec3> full=evaluated;FrameStats stats;SettleCounters settle=settle_full(full,p,repaired_mask);stats.repaired_strands=settle.repaired;stats.settled_strands=settle.settled;stats.failed_strands=settle.failed;stats.length_bad_before=settle.length_bad;stats.length_bad_rods_before=settle.length_bad_rods;stats.folded_before=settle.folded;stats.shape_bad_before=settle.shape_bad;stats.rough_bad_before=settle.rough_bad;stats.collision_bad_before=settle.collision_bad;stats.shape_bad_after=settle.shape_bad_after;stats.collision_bad_after=settle.collision_bad_after;stats.max_settle_iterations=settle.max_iterations;stats.max_length_error_mm=settle.max_length_error*1000.0f;stats.max_final_length_error_mm=settle.max_final_length_error*1000.0f;stats.max_angle_deg=settle.max_angle;return {std::move(full),stats};
    }

    void sync_external(const std::vector<Vec3>& evaluated,const std::vector<int>& repaired_mask) {
        if(evaluated.size()!=full_rest_positions_.size()||static_cast<int>(repaired_mask.size())!=n_full_)throw std::runtime_error("sync topology changed");for(int g=0;g<n_guides_;++g){const int fs=guides_[g];for(int j=0;j<pps_;++j){pos_[g*pps_+j]=predicted_[g*pps_+j]=previous_eval_guides_[g*pps_+j]=evaluated[point_index(fs,j)];if(repaired_mask[fs])vel_[g*pps_+j]={};}}
    }
};

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

std::vector<Vec3> read_positions(const py::handle& value, const char* name) {
    FloatArray array = FloatArray::ensure(value);
    if (!array) throw py::type_error(std::string(name) + " must be convertible to a contiguous float32 array");
    const py::buffer_info info = array.request();
    std::size_t count = 0;
    if (info.ndim == 2 && info.shape[1] == 3) count = static_cast<std::size_t>(info.shape[0]);
    else if (info.ndim == 3 && info.shape[2] == 3) count = static_cast<std::size_t>(info.shape[0] * info.shape[1]);
    else throw py::value_error(std::string(name) + " must have shape (N, 3) or (strands, points, 3)");
    const float* source = static_cast<const float*>(info.ptr);
    std::vector<Vec3> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        if(!std::isfinite(source[3*i])||!std::isfinite(source[3*i+1])||!std::isfinite(source[3*i+2]))throw py::value_error(std::string(name)+" contains NaN or infinity");
        result[i] = {source[3*i], source[3*i+1], source[3*i+2]};
    }
    return result;
}

std::vector<int> read_indices(const py::handle& value, const char* name) {
    IntArray array = IntArray::ensure(value);
    if (!array) throw py::type_error(std::string(name) + " must be convertible to a contiguous int32 array");
    const py::buffer_info info = array.request();
    std::size_t count = 0;
    if (info.ndim == 1) count = static_cast<std::size_t>(info.shape[0]);
    else if (info.ndim == 2 && info.shape[1] == 3) count = static_cast<std::size_t>(info.shape[0] * 3);
    else throw py::value_error(std::string(name) + " must have shape (T, 3) or (3*T,)");
    const int* source = static_cast<const int*>(info.ptr);
    return {source, source + count};
}

py::array_t<float> write_positions(const std::vector<Vec3>& positions, int strands, int points_per_strand) {
    if (positions.size() != static_cast<std::size_t>(strands) * points_per_strand)
        throw std::runtime_error("native result topology is inconsistent");
    py::array_t<float> result({strands, points_per_strand, 3});
    float* destination = result.mutable_data();
    for (std::size_t i = 0; i < positions.size(); ++i) {
        destination[3*i] = positions[i].x;
        destination[3*i+1] = positions[i].y;
        destination[3*i+2] = positions[i].z;
    }
    return result;
}

py::array_t<std::int32_t> write_mask(const std::vector<int>& mask) {
    py::array_t<std::int32_t> result(mask.size());
    std::int32_t* destination = result.mutable_data();
    for (std::size_t i = 0; i < mask.size(); ++i) destination[i] = static_cast<std::int32_t>(mask[i]);
    return result;
}

template <typename T>
void read_parameter(const py::dict& values, const char* name, T& destination) {
    if (values.contains(name)) destination = py::cast<T>(values[name]);
}

Parameters read_parameters(const py::object& value) {
    Parameters p;
    if (value.is_none()) return p;
    if (!py::isinstance<py::dict>(value)) throw py::type_error("parameters must be a dict or None");
    const py::dict values = py::reinterpret_borrow<py::dict>(value);
    if (values.contains("gravity")) {
        FloatArray gravity = FloatArray::ensure(values["gravity"]);
        if (!gravity || gravity.size() != 3) throw py::value_error("gravity must contain three floats");
        const float* v = gravity.data(); p.gravity = {v[0], v[1], v[2]};
    }
    read_parameter(values,"damping",p.damping);read_parameter(values,"internal_damping",p.internal_damping);
    read_parameter(values,"max_velocity",p.max_velocity);read_parameter(values,"iterations",p.iterations);
    read_parameter(values,"stretch_stiffness",p.stretch_stiffness);read_parameter(values,"bend_stiffness",p.bend_stiffness);
    read_parameter(values,"collision_margin",p.collision_margin);read_parameter(values,"collision_search",p.collision_search);
    read_parameter(values,"collision_max_correction",p.collision_max_correction);read_parameter(values,"collision_response",p.collision_response);
    read_parameter(values,"collision_velocity_damping",p.collision_velocity_damping);read_parameter(values,"collision_smoothing",p.collision_smoothing);
    read_parameter(values,"collision_passes",p.collision_passes);read_parameter(values,"post_collision_iterations",p.post_collision_iterations);
    read_parameter(values,"max_move_per_substep",p.max_move_per_substep);read_parameter(values,"max_substeps",p.max_substeps);
    read_parameter(values,"keep_length",p.keep_length);read_parameter(values,"adaptive_root_lock",p.adaptive_root_lock);
    read_parameter(values,"settle_iterations",p.settle_iterations);read_parameter(values,"settle_relaxation",p.settle_relaxation);
    read_parameter(values,"groom_strength",p.groom_strength);read_parameter(values,"repair_strength",p.repair_strength);
    read_parameter(values,"length_tolerance",p.length_tolerance);read_parameter(values,"angle_change_limit_deg",p.angle_change_limit_deg);
    read_parameter(values,"fold_limit_deg",p.fold_limit_deg);read_parameter(values,"roughness_factor",p.roughness_factor);
    read_parameter(values,"settle_min_improvement",p.settle_min_improvement);read_parameter(values,"settle_stagnation",p.settle_stagnation);
    read_parameter(values,"collision_smooth_passes",p.collision_smooth_passes);read_parameter(values,"openmp_threads",p.openmp_threads);
    if(!std::isfinite(p.gravity.x)||!std::isfinite(p.gravity.y)||!std::isfinite(p.gravity.z)||
       !std::isfinite(p.damping)||!std::isfinite(p.internal_damping)||!std::isfinite(p.max_velocity)||
       !std::isfinite(p.stretch_stiffness)||!std::isfinite(p.bend_stiffness)||!std::isfinite(p.collision_margin)||
       !std::isfinite(p.collision_search)||!std::isfinite(p.collision_max_correction)||!std::isfinite(p.collision_response)||
       !std::isfinite(p.collision_velocity_damping)||!std::isfinite(p.collision_smoothing)||!std::isfinite(p.max_move_per_substep)||
       !std::isfinite(p.settle_relaxation)||!std::isfinite(p.groom_strength)||!std::isfinite(p.repair_strength)||
       !std::isfinite(p.length_tolerance)||!std::isfinite(p.angle_change_limit_deg)||!std::isfinite(p.fold_limit_deg)||
       !std::isfinite(p.roughness_factor)||!std::isfinite(p.settle_min_improvement))
        throw py::value_error("parameters contain NaN or infinity");
    p.damping=clampf(p.damping,0.0f,0.999f);p.internal_damping=clampf(p.internal_damping,0.0f,0.5f);p.max_velocity=std::max(0.0f,p.max_velocity);
    p.iterations=std::max(1,p.iterations);p.stretch_stiffness=std::max(1.0e-8f,p.stretch_stiffness);p.bend_stiffness=std::max(0.0f,p.bend_stiffness);
    p.collision_margin=std::max(0.0f,p.collision_margin);p.collision_search=std::max(p.collision_margin,p.collision_search);p.collision_max_correction=std::max(1.0e-8f,p.collision_max_correction);
    p.collision_response=clampf(p.collision_response,0.0f,1.0f);p.collision_velocity_damping=clampf(p.collision_velocity_damping,0.0f,1.0f);p.collision_smoothing=clampf(p.collision_smoothing,0.0f,1.0f);
    p.collision_passes=std::max(1,p.collision_passes);p.post_collision_iterations=std::max(0,p.post_collision_iterations);p.max_move_per_substep=std::max(1.0e-7f,p.max_move_per_substep);p.max_substeps=std::max(1,p.max_substeps);
    p.settle_iterations=std::max(1,p.settle_iterations);p.settle_relaxation=clampf(p.settle_relaxation,0.01f,1.0f);p.groom_strength=clampf(p.groom_strength,0.0f,1.0f);p.repair_strength=clampf(p.repair_strength,0.0f,1.0f);
    p.length_tolerance=std::max(0.0f,p.length_tolerance);p.angle_change_limit_deg=clampf(p.angle_change_limit_deg,0.0f,180.0f);p.fold_limit_deg=clampf(p.fold_limit_deg,1.0f,180.0f);p.roughness_factor=std::max(0.0f,p.roughness_factor);
    p.settle_stagnation=std::max(1,p.settle_stagnation);p.settle_min_improvement=std::max(0.0f,p.settle_min_improvement);p.collision_smooth_passes=std::max(0,p.collision_smooth_passes);
    omp_set_num_threads(p.openmp_threads > 0 ? std::min(p.openmp_threads, omp_get_num_procs()) : omp_get_num_procs());
    return p;
}

std::optional<HeadFrame> read_head_frame(const py::object& value) {
    if (value.is_none()) return std::nullopt;
    if (!py::isinstance<py::dict>(value)) throw py::type_error("head_frame must be a dict or None");
    const py::dict values = py::reinterpret_borrow<py::dict>(value);
    auto vector_value = [&](const char* name) {
        if (!values.contains(name)) throw py::value_error(std::string("head_frame is missing ") + name);
        FloatArray array = FloatArray::ensure(values[name]);
        if (!array || array.size() != 3) throw py::value_error(std::string("head_frame.") + name + " must contain three floats");
        const float* v = array.data(); return Vec3{v[0],v[1],v[2]};
    };
    HeadFrame result;result.base=vector_value("base");result.up=vector_value("up");result.tip=vector_value("tip");
    if (values.contains("ear_offset")) result.ear_offset=py::cast<float>(values["ear_offset"]);
    const auto finite_vec=[](const Vec3& v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);};
    if(!finite_vec(result.base)||!finite_vec(result.up)||!finite_vec(result.tip)||!std::isfinite(result.ear_offset))
        throw py::value_error("head_frame contains NaN or infinity");
    result.up=normalize_or(result.up,{0,0,1});
    return result;
}

py::dict write_stats(const FrameStats& stats) {
    py::dict result;
    result["substeps"]=stats.substeps;result["hits"]=stats.hits;result["repaired_strands"]=stats.repaired_strands;
    result["settled_strands"]=stats.settled_strands;result["failed_strands"]=stats.failed_strands;
    result["length_bad_before"]=stats.length_bad_before;result["length_bad_rods_before"]=stats.length_bad_rods_before;result["folded_before"]=stats.folded_before;
    result["shape_bad_before"]=stats.shape_bad_before;result["rough_bad_before"]=stats.rough_bad_before;result["collision_bad_before"]=stats.collision_bad_before;
    result["shape_bad_after"]=stats.shape_bad_after;result["collision_bad_after"]=stats.collision_bad_after;
    result["max_settle_iterations"]=stats.max_settle_iterations;result["adaptive_lock_max_points"]=stats.adaptive_lock_max_points;
    result["adaptive_lock_strands"]=stats.adaptive_lock_strands;result["auto_move_mm"]=stats.auto_move_mm;
    result["max_length_error_mm"]=stats.max_length_error_mm;result["max_final_length_error_mm"]=stats.max_final_length_error_mm;result["max_angle_deg"]=stats.max_angle_deg;
    return result;
}

py::dict write_result(std::pair<std::vector<Vec3>,FrameStats>&& native_result, const std::vector<int>& mask, const Simulator& simulator) {
    py::dict result;
    result["positions"] = write_positions(native_result.first, simulator.strand_count(), simulator.points_per_strand());
    result["repaired_mask"] = write_mask(mask);
    result["stats"] = write_stats(native_result.second);
    return result;
}

} // namespace yurameki

PYBIND11_MODULE(_yurameki_native_0_3_0, module) {
    using namespace yurameki;
    module.doc() = "Yurameki C++/OpenMP rod simulation, collision, grooming, and convergence core";
    module.attr("__version__") = "0.3.0";
    module.def("openmp_enabled", [](){ return true; });
    module.def("max_threads", [](){ return omp_get_num_procs(); });

    py::class_<Simulator>(module, "Simulator", py::module_local())
        .def(py::init([](const py::object& initial, const py::object& frame1_rest, int points_per_strand,
                         int root_locked_points, float particle_mass, int guide_decimation) {
            std::vector<Vec3> initial_positions = read_positions(initial,"initial");
            std::vector<Vec3> rest_positions = read_positions(frame1_rest,"frame1_rest");
            std::unique_ptr<Simulator> result;
            {
                py::gil_scoped_release release;
                result = std::make_unique<Simulator>(initial_positions,rest_positions,points_per_strand,root_locked_points,particle_mass,guide_decimation);
            }
            return result;
        }), py::arg("initial"), py::arg("frame1_rest"), py::arg("points_per_strand"),
            py::arg("root_locked_points"), py::arg("particle_mass"), py::arg("guide_decimation")=1)
        .def_property_readonly("strand_count",&Simulator::strand_count)
        .def_property_readonly("points_per_strand",&Simulator::points_per_strand)
        .def_property_readonly("guide_count",&Simulator::guide_count)
        .def("prime_head",[](Simulator& simulator,const py::object& head_frame){simulator.prime_head(read_head_frame(head_frame));},py::arg("head_frame")=py::none())
        .def("audit",[](Simulator& simulator,const py::object& evaluated,const py::object& body_vertices,const py::object& body_triangles,
                         const py::object& clothes_vertices,const py::object& clothes_triangles,const py::object& parameters) {
            std::vector<Vec3> target=read_positions(evaluated,"evaluated");std::vector<Vec3> body=read_positions(body_vertices,"body_vertices");
            std::vector<int> body_index=read_indices(body_triangles,"body_triangles");std::vector<Vec3> clothes=read_positions(clothes_vertices,"clothes_vertices");
            std::vector<int> clothes_index=read_indices(clothes_triangles,"clothes_triangles");Parameters p=read_parameters(parameters);FrameStats stats;
            {py::gil_scoped_release release;stats=simulator.audit(target,body,body_index,clothes,clothes_index,p);}return write_stats(stats);
        },py::arg("evaluated"),py::arg("body_vertices"),py::arg("body_triangles"),py::arg("clothes_vertices"),py::arg("clothes_triangles"),py::arg("parameters")=py::none())
        .def("frame",[](Simulator& simulator,const py::object& evaluated,const py::object& body_vertices,const py::object& body_triangles,
                         const py::object& clothes_vertices,const py::object& clothes_triangles,float dt_frame,const py::object& parameters,const py::object& head_frame) {
            std::vector<Vec3> target=read_positions(evaluated,"evaluated");std::vector<Vec3> body=read_positions(body_vertices,"body_vertices");
            std::vector<int> body_index=read_indices(body_triangles,"body_triangles");std::vector<Vec3> clothes=read_positions(clothes_vertices,"clothes_vertices");
            std::vector<int> clothes_index=read_indices(clothes_triangles,"clothes_triangles");Parameters p=read_parameters(parameters);
            std::optional<HeadFrame> head=read_head_frame(head_frame);std::vector<int> mask;std::pair<std::vector<Vec3>,FrameStats> result;
            {py::gil_scoped_release release;result=simulator.frame(target,body,body_index,clothes,clothes_index,dt_frame,p,head,mask);}
            return write_result(std::move(result),mask,simulator);
        },py::arg("evaluated"),py::arg("body_vertices"),py::arg("body_triangles"),py::arg("clothes_vertices"),py::arg("clothes_triangles"),
          py::arg("dt_frame"),py::arg("parameters")=py::none(),py::arg("head_frame")=py::none())
        .def("settle_external",[](Simulator& simulator,const py::object& evaluated,const py::object& parameters) {
            std::vector<Vec3> target=read_positions(evaluated,"evaluated");Parameters p=read_parameters(parameters);std::vector<int> mask;
            std::pair<std::vector<Vec3>,FrameStats> result;{py::gil_scoped_release release;result=simulator.settle_external(target,p,mask);}
            return write_result(std::move(result),mask,simulator);
        },py::arg("evaluated"),py::arg("parameters")=py::none())
        .def("sync_external",[](Simulator& simulator,const py::object& evaluated,const py::object& repaired_mask) {
            std::vector<Vec3> target=read_positions(evaluated,"evaluated");std::vector<int> mask=read_indices(repaired_mask,"repaired_mask");
            py::gil_scoped_release release;simulator.sync_external(target,mask);
        },py::arg("evaluated"),py::arg("repaired_mask"));
}
