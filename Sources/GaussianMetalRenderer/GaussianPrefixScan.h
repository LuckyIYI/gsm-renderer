// GaussianPrefixScan.h - Shared prefix scan templates for Gaussian splatting
// Used by both global radix sort and local sort pipelines

#ifndef GAUSSIAN_PREFIX_SCAN_H
#define GAUSSIAN_PREFIX_SCAN_H

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// PREFIX SCAN CONFIGURATION
// =============================================================================

#define TILE_PREFIX_BLOCK_SIZE 256
#define TILE_PREFIX_GRAIN_SIZE 4

// =============================================================================
// DISCONTINUITY FLAGS
// =============================================================================

template <ushort BLOCK_SIZE, typename T>
static uchar FlagHeadDiscontinuity(const T value, threadgroup T* shared, const ushort local_id) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar result = (local_id == 0) ? 1 : shared[local_id] != shared[local_id - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

template <ushort BLOCK_SIZE, typename T>
static uchar FlagTailDiscontinuity(const T value, threadgroup T* shared, const ushort local_id) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar result = (local_id == BLOCK_SIZE - 1) ? 1 : shared[local_id] != shared[local_id + 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// =============================================================================
// GLOBAL MEMORY LOAD/STORE HELPERS
// =============================================================================

template<ushort LENGTH, typename T>
static void LoadStripedLocalFromGlobal(thread T (&value)[LENGTH],
                                        const device T* input_data,
                                        const ushort local_id,
                                        const ushort local_size) {
    for (ushort i = 0; i < LENGTH; i++) {
        value[i] = input_data[local_id + i * local_size];
    }
}

template<ushort LENGTH, typename T>
static void LoadStripedLocalFromGlobal(thread T (&value)[LENGTH],
                                        const device T* input_data,
                                        const ushort local_id,
                                        const ushort local_size,
                                        const uint n,
                                        const T substitution_value) {
    for (ushort i = 0; i < LENGTH; i++) {
        value[i] = (local_id + i * local_size < n) ? input_data[local_id + i * local_size] : substitution_value;
    }
}

template<ushort GRAIN_SIZE, typename T>
static void LoadBlockedLocalFromGlobal(thread T (&value)[GRAIN_SIZE],
                                        const device T* input_data,
                                        const ushort local_id) {
    for (ushort i = 0; i < GRAIN_SIZE; i++) {
        value[i] = input_data[local_id * GRAIN_SIZE + i];
    }
}

template<ushort GRAIN_SIZE, typename T>
static void LoadBlockedLocalFromGlobal(thread T (&value)[GRAIN_SIZE],
                                        const device T* input_data,
                                        const ushort local_id,
                                        const uint n,
                                        const T substitution_value) {
    for (ushort i = 0; i < GRAIN_SIZE; i++) {
        value[i] = (local_id * GRAIN_SIZE + i < n) ? input_data[local_id * GRAIN_SIZE + i] : substitution_value;
    }
}

template<ushort GRAIN_SIZE, typename T>
static void StoreBlockedLocalToGlobal(device T *output_data,
                                       thread const T (&value)[GRAIN_SIZE],
                                       const ushort local_id) {
    for (ushort i = 0; i < GRAIN_SIZE; i++) {
        output_data[local_id * GRAIN_SIZE + i] = value[i];
    }
}

template<ushort GRAIN_SIZE, typename T>
static void StoreBlockedLocalToGlobal(device T *output_data,
                                       thread const T (&value)[GRAIN_SIZE],
                                       const ushort local_id,
                                       const uint n) {
    for (ushort i = 0; i < GRAIN_SIZE; i++) {
        if (local_id * GRAIN_SIZE + i < n) {
            output_data[local_id * GRAIN_SIZE + i] = value[i];
        }
    }
}

// =============================================================================
// THREAD-LOCAL PREFIX SUM
// =============================================================================

template<ushort LENGTH, typename T>
static inline T ThreadPrefixInclusiveSum(thread T (&values)[LENGTH]) {
    for (ushort i = 1; i < LENGTH; i++) {
        values[i] += values[i - 1];
    }
    return values[LENGTH - 1];
}

template<ushort LENGTH, typename T>
static inline T ThreadPrefixInclusiveSum(threadgroup T* values) {
    for (ushort i = 1; i < LENGTH; i++) {
        values[i] += values[i - 1];
    }
    return values[LENGTH - 1];
}

template<ushort LENGTH, typename T>
static inline T ThreadPrefixExclusiveSum(thread T (&values)[LENGTH]) {
    T inclusive_prefix = ThreadPrefixInclusiveSum<LENGTH>(values);
    for (ushort i = LENGTH - 1; i > 0; i--) {
        values[i] = values[i - 1];
    }
    values[0] = 0;
    return inclusive_prefix;
}

template<ushort LENGTH, typename T>
static inline T ThreadPrefixExclusiveSum(threadgroup T* values) {
    T inclusive_prefix = ThreadPrefixInclusiveSum<LENGTH>(values);
    for (ushort i = LENGTH - 1; i > 0; i--) {
        values[i] = values[i - 1];
    }
    values[0] = 0;
    return inclusive_prefix;
}

template<ushort LENGTH, typename T>
static inline void ThreadUniformAdd(thread T (&values)[LENGTH], T uni) {
    for (ushort i = 0; i < LENGTH; i++) {
        values[i] += uni;
    }
}

template<ushort LENGTH, typename T>
static inline void ThreadUniformAdd(threadgroup T* values, T uni) {
    for (ushort i = 0; i < LENGTH; i++) {
        values[i] += uni;
    }
}

// =============================================================================
// THREADGROUP PREFIX SCAN ALGORITHMS
// =============================================================================

/// Blelloch (work-efficient) prefix exclusive sum
template<ushort BLOCK_SIZE, typename T>
static T ThreadgroupBlellochPrefixExclusiveSum(T value, threadgroup T* sdata, const ushort lid) {
    sdata[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort ai = 2 * lid + 1;
    const ushort bi = 2 * lid + 2;

    // Up-sweep (reduce)
    if (BLOCK_SIZE >=    2) { if (lid < (BLOCK_SIZE >>  1)) { sdata[   1 * bi - 1] += sdata[   1 * ai - 1]; } if ((BLOCK_SIZE >>  0) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=    4) { if (lid < (BLOCK_SIZE >>  2)) { sdata[   2 * bi - 1] += sdata[   2 * ai - 1]; } if ((BLOCK_SIZE >>  1) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=    8) { if (lid < (BLOCK_SIZE >>  3)) { sdata[   4 * bi - 1] += sdata[   4 * ai - 1]; } if ((BLOCK_SIZE >>  2) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=   16) { if (lid < (BLOCK_SIZE >>  4)) { sdata[   8 * bi - 1] += sdata[   8 * ai - 1]; } if ((BLOCK_SIZE >>  3) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=   32) { if (lid < (BLOCK_SIZE >>  5)) { sdata[  16 * bi - 1] += sdata[  16 * ai - 1]; } if ((BLOCK_SIZE >>  4) > 32) threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=   64) { if (lid < (BLOCK_SIZE >>  6)) { sdata[  32 * bi - 1] += sdata[  32 * ai - 1]; } }
    if (BLOCK_SIZE >=  128) { if (lid < (BLOCK_SIZE >>  7)) { sdata[  64 * bi - 1] += sdata[  64 * ai - 1]; } }
    if (BLOCK_SIZE >=  256) { if (lid < (BLOCK_SIZE >>  8)) { sdata[ 128 * bi - 1] += sdata[ 128 * ai - 1]; } }
    if (BLOCK_SIZE >=  512) { if (lid < (BLOCK_SIZE >>  9)) { sdata[ 256 * bi - 1] += sdata[ 256 * ai - 1]; } }
    if (BLOCK_SIZE >= 1024) { if (lid < (BLOCK_SIZE >> 10)) { sdata[ 512 * bi - 1] += sdata[ 512 * ai - 1]; } }

    if (lid == 0) {
        sdata[BLOCK_SIZE - 1] = 0;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Down-sweep
    if (BLOCK_SIZE >=    2) {
        if (lid <    1) {
            sdata[(BLOCK_SIZE >>  1) * bi - 1] += sdata[(BLOCK_SIZE >>  1) * ai - 1];
            sdata[(BLOCK_SIZE >>  1) * ai - 1] = sdata[(BLOCK_SIZE >>  1) * bi - 1] - sdata[(BLOCK_SIZE >>  1) * ai - 1];
        }
    }
    if (BLOCK_SIZE >=    4) { if (lid <    2) { sdata[(BLOCK_SIZE >>  2) * bi - 1] += sdata[(BLOCK_SIZE >>  2) * ai - 1]; sdata[(BLOCK_SIZE >>  2) * ai - 1] = sdata[(BLOCK_SIZE >>  2) * bi - 1] - sdata[(BLOCK_SIZE >>  2) * ai - 1]; } }
    if (BLOCK_SIZE >=    8) { if (lid <    4) { sdata[(BLOCK_SIZE >>  3) * bi - 1] += sdata[(BLOCK_SIZE >>  3) * ai - 1]; sdata[(BLOCK_SIZE >>  3) * ai - 1] = sdata[(BLOCK_SIZE >>  3) * bi - 1] - sdata[(BLOCK_SIZE >>  3) * ai - 1]; } }
    if (BLOCK_SIZE >=   16) { if (lid <    8) { sdata[(BLOCK_SIZE >>  4) * bi - 1] += sdata[(BLOCK_SIZE >>  4) * ai - 1]; sdata[(BLOCK_SIZE >>  4) * ai - 1] = sdata[(BLOCK_SIZE >>  4) * bi - 1] - sdata[(BLOCK_SIZE >>  4) * ai - 1]; } }
    if (BLOCK_SIZE >=   32) { if (lid <   16) { sdata[(BLOCK_SIZE >>  5) * bi - 1] += sdata[(BLOCK_SIZE >>  5) * ai - 1]; sdata[(BLOCK_SIZE >>  5) * ai - 1] = sdata[(BLOCK_SIZE >>  5) * bi - 1] - sdata[(BLOCK_SIZE >>  5) * ai - 1]; } }
    if (BLOCK_SIZE >=   64) { if (lid <   32) { sdata[(BLOCK_SIZE >>  6) * bi - 1] += sdata[(BLOCK_SIZE >>  6) * ai - 1]; sdata[(BLOCK_SIZE >>  6) * ai - 1] = sdata[(BLOCK_SIZE >>  6) * bi - 1] - sdata[(BLOCK_SIZE >>  6) * ai - 1]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=  128) { if (lid <   64) { sdata[(BLOCK_SIZE >>  7) * bi - 1] += sdata[(BLOCK_SIZE >>  7) * ai - 1]; sdata[(BLOCK_SIZE >>  7) * ai - 1] = sdata[(BLOCK_SIZE >>  7) * bi - 1] - sdata[(BLOCK_SIZE >>  7) * ai - 1]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=  256) { if (lid <  128) { sdata[(BLOCK_SIZE >>  8) * bi - 1] += sdata[(BLOCK_SIZE >>  8) * ai - 1]; sdata[(BLOCK_SIZE >>  8) * ai - 1] = sdata[(BLOCK_SIZE >>  8) * bi - 1] - sdata[(BLOCK_SIZE >>  8) * ai - 1]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >=  512) { if (lid <  256) { sdata[(BLOCK_SIZE >>  9) * bi - 1] += sdata[(BLOCK_SIZE >>  9) * ai - 1]; sdata[(BLOCK_SIZE >>  9) * ai - 1] = sdata[(BLOCK_SIZE >>  9) * bi - 1] - sdata[(BLOCK_SIZE >>  9) * ai - 1]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (BLOCK_SIZE >= 1024) { if (lid <  512) { sdata[(BLOCK_SIZE >> 10) * bi - 1] += sdata[(BLOCK_SIZE >> 10) * ai - 1]; sdata[(BLOCK_SIZE >> 10) * ai - 1] = sdata[(BLOCK_SIZE >> 10) * bi - 1] - sdata[(BLOCK_SIZE >> 10) * ai - 1]; } threadgroup_barrier(mem_flags::mem_threadgroup); }

    return sdata[lid];
}

/// Raking prefix exclusive sum (SIMD-optimized)
template<ushort BLOCK_SIZE, typename T>
static T ThreadgroupRakingPrefixExclusiveSum(T value, threadgroup T* shared, const ushort lid) {
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < 32) {
        const short values_per_thread = BLOCK_SIZE / 32;
        const short first_index = lid * values_per_thread;
        for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
            shared[i] += shared[i - 1];
        }
        T partial_sum = shared[first_index + values_per_thread - 1];
        for (short i = first_index + values_per_thread - 1; i > first_index; i--) {
            shared[i] = shared[i - 1];
        }
        shared[first_index] = 0;

        T prefix = simd_prefix_exclusive_sum(partial_sum);

        for (short i = first_index; i < first_index + values_per_thread; i++) {
            shared[i] += prefix;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return shared[lid];
}

#endif // GAUSSIAN_PREFIX_SCAN_H
