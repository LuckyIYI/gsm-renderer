// ClusterCullShaders.metal - Skip-based cluster culling for Morton-sorted Gaussians
// ULTRA-SIMPLE: No compaction, just visibility buffer for projection to check
// Pass 1: Precompute AABBs (once at load time)
// Pass 2: Fast frustum cull → write clusterVisible[clusterId] = 0 or 1
// Projection kernel checks visibility and early-returns for culled clusters

#include <metal_stdlib>
#include "BridgingTypes.h"
using namespace metal;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

inline float computeGaussianRadius(float3 scale) {
    return max(scale.x, max(scale.y, scale.z)) * 3.0f;
}

// Test if AABB is at least partially inside frustum (p-vertex method)
inline bool aabbInFrustum(float3 aabbMin, float3 aabbMax, constant FrustumPlanes& frustum) {
    #define TEST_PLANE(i) { \
        float4 plane = frustum.planes[i]; \
        float3 pVertex = select(aabbMin, aabbMax, plane.xyz >= 0); \
        if (dot(plane.xyz, pVertex) + plane.w < 0) return false; \
    }
    TEST_PLANE(0); TEST_PLANE(1); TEST_PLANE(2);
    TEST_PLANE(3); TEST_PLANE(4); TEST_PLANE(5);
    #undef TEST_PLANE
    return true;
}

// =============================================================================
// KERNEL 1: PRE-COMPUTE CLUSTER AABBs (RUN ONCE AT LOAD TIME)
// =============================================================================

template <typename PackedWorldT>
kernel void precomputeClusterAABBs(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    device ClusterAABB* clusterAABBs [[buffer(1)]],
    constant ClusterCullParams& params [[buffer(2)]],
    uint clusterId [[thread_position_in_grid]]
) {
    if (clusterId >= params.clusterCount) return;

    uint startIdx = clusterId * params.gaussiansPerCluster;
    uint endIdx = min(startIdx + params.gaussiansPerCluster, params.totalGaussians);

    if (startIdx >= params.totalGaussians) {
        clusterAABBs[clusterId].minBounds = float3(INFINITY);
        clusterAABBs[clusterId].maxBounds = float3(-INFINITY);
        return;
    }

    PackedWorldT g0 = worldGaussians[startIdx];
    float3 pos0 = float3(g0.px, g0.py, g0.pz);
    float radius0 = computeGaussianRadius(float3(g0.sx, g0.sy, g0.sz));
    float3 minBounds = pos0 - radius0;
    float3 maxBounds = pos0 + radius0;

    for (uint i = startIdx + 1; i < endIdx; i++) {
        PackedWorldT g = worldGaussians[i];
        float3 pos = float3(g.px, g.py, g.pz);
        float radius = computeGaussianRadius(float3(g.sx, g.sy, g.sz));
        minBounds = min(minBounds, pos - radius);
        maxBounds = max(maxBounds, pos + radius);
    }

    clusterAABBs[clusterId].minBounds = minBounds;
    clusterAABBs[clusterId].maxBounds = maxBounds;
}

template [[host_name("precomputeClusterAABBsFloat")]]
kernel void precomputeClusterAABBs<PackedWorldGaussian>(
    const device PackedWorldGaussian*, device ClusterAABB*, constant ClusterCullParams&, uint);

template [[host_name("precomputeClusterAABBsHalf")]]
kernel void precomputeClusterAABBs<PackedWorldGaussianHalf>(
    const device PackedWorldGaussianHalf*, device ClusterAABB*, constant ClusterCullParams&, uint);

// =============================================================================
// KERNEL 2: SIMPLE VISIBILITY CULL (RUNTIME)
// =============================================================================
// Ultra-simple: check AABB vs frustum, write 0 or 1 to visibility buffer.
// Projection kernel will check this and early-return for culled clusters.

kernel void cullClustersSimple(
    const device ClusterAABB* clusterAABBs [[buffer(0)]],
    device uchar* clusterVisible [[buffer(1)]],  // 1 byte per cluster
    constant FrustumPlanes& frustum [[buffer(2)]],
    constant uint& clusterCount [[buffer(3)]],
    uint clusterId [[thread_position_in_grid]]
) {
    if (clusterId >= clusterCount) return;

    ClusterAABB aabb = clusterAABBs[clusterId];

    // Invalid cluster check
    bool valid = aabb.minBounds.x <= aabb.maxBounds.x;
    bool visible = valid && aabbInFrustum(aabb.minBounds, aabb.maxBounds, frustum);

    clusterVisible[clusterId] = visible ? 1 : 0;
}
