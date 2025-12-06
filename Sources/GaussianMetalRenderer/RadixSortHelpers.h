// RadixSortHelpers.h - Radix sort helper functions and templates
// Used by the global radix sort pipeline
// All functions use snake_case to match Metal's native SIMD style

#ifndef RADIX_SORT_HELPERS_H
#define RADIX_SORT_HELPERS_H

#include <metal_stdlib>
#include "BridgingTypes.h"
using namespace metal;

// =============================================================================
// RADIX SORT CONSTANTS
// =============================================================================

#define RADIX_BLOCK_SIZE 256
#define RADIX_HISTOGRAM_SCAN_BLOCK_SIZE 256
#define RADIX_GRAIN_SIZE 4
#define RADIX_BUCKETS 256
#define RADIX_BITS_PER_PASS 8
#define RADIX_NUM_PASSES (64 / RADIX_BITS_PER_PASS)

#define RADIX_SCAN_TYPE_INCLUSIVE (0)
#define RADIX_SCAN_TYPE_EXCLUSIVE (1)

// =============================================================================
// RADIX SORT TYPES
// =============================================================================

using RadixKeyType = uint;  // 32-bit key: [tile:16][depth:16]

struct RadixKeyPayload {
    RadixKeyType key;
    uint payload;
};

// =============================================================================
// BINARY OPERATORS
// =============================================================================

template <typename T>
struct radix_sum_op {
    inline T operator()(thread const T& a, thread const T& b) const { return a + b; }
    inline T operator()(threadgroup const T& a, thread const T& b) const { return a + b; }
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const { return a + b; }
    inline T operator()(volatile threadgroup const T& a, volatile threadgroup const T& b) const { return a + b; }
    constexpr T identity() { return static_cast<T>(0); }
};

template <typename T>
struct radix_max_op {
    inline T operator()(thread const T& a, thread const T& b) const { return max(a, b); }
    inline T operator()(threadgroup const T& a, thread const T& b) const { return max(a, b); }
    inline T operator()(threadgroup const T& a, threadgroup const T& b) const { return max(a, b); }
    inline T operator()(volatile threadgroup const T& a, volatile threadgroup const T& b) const { return max(a, b); }
    constexpr T identity() { return metal::numeric_limits<T>::min(); }
};

// =============================================================================
// BIT MANIPULATION HELPERS
// =============================================================================

/// Compute log2(radix) - number of bits per digit
static constexpr ushort radix_to_bits(ushort n) {
    return (n - 1 < 2) ? 1 :
           (n - 1 < 4) ? 2 :
           (n - 1 < 8) ? 3 :
           (n - 1 < 16) ? 4 :
           (n - 1 < 32) ? 5 :
           (n - 1 < 64) ? 6 :
           (n - 1 < 128) ? 7 :
           (n - 1 < 256) ? 8 :
           (n - 1 < 512) ? 9 :
           (n - 1 < 1024) ? 10 :
           (n - 1 < 2048) ? 11 :
           (n - 1 < 4096) ? 12 :
           (n - 1 < 8192) ? 13 :
           (n - 1 < 16384) ? 14 :
           (n - 1 < 32768) ? 15 : 0;
}

/// Extract digit from value at given bit position
template <ushort R, typename T>
static inline ushort value_to_key_at_bit(T value, ushort current_bit) {
    return (value >> current_bit) & (R - 1);
}

/// Extract digit from value at given digit position
static inline ushort value_to_key_at_digit(RadixKeyType value, ushort current_digit) {
    ushort bits_to_shift = radix_to_bits(RADIX_BUCKETS) * current_digit;
    return value_to_key_at_bit<RADIX_BUCKETS>(value, bits_to_shift);
}

/// Extract digit from key-payload pair
template <ushort R>
static inline ushort value_to_key_at_bit(RadixKeyPayload value, ushort current_bit) {
    return value_to_key_at_bit<R>(value.key, current_bit);
}

// =============================================================================
// THREAD-LEVEL SCAN HELPERS
// =============================================================================

/// Thread-local scan with configurable scan type (inclusive/exclusive)
template <ushort LENGTH, int SCAN_TYPE, typename BinaryOp, typename T>
static inline T radix_thread_scan(threadgroup T* values, BinaryOp Op) {
    for (ushort i = 1; i < LENGTH; i++) {
        values[i] = Op(values[i], values[i - 1]);
    }
    T result = values[LENGTH - 1];
    if (SCAN_TYPE == RADIX_SCAN_TYPE_EXCLUSIVE) {
        for (ushort i = LENGTH - 1; i > 0; i--) {
            values[i] = values[i - 1];
        }
        values[0] = 0;
    }
    return result;
}

/// SIMD-level scan with configurable scan type
template <int SCAN_TYPE, typename BinaryOp, typename T>
static inline T radix_simdgroup_scan(T value, ushort local_id, BinaryOp Op) {
    const ushort lane_id = local_id % 32;
    T temp = simd_shuffle_up(value, 1);
    if (lane_id >= 1) value = Op(value, temp);
    temp = simd_shuffle_up(value, 2);
    if (lane_id >= 2) value = Op(value, temp);
    temp = simd_shuffle_up(value, 4);
    if (lane_id >= 4) value = Op(value, temp);
    temp = simd_shuffle_up(value, 8);
    if (lane_id >= 8) value = Op(value, temp);
    temp = simd_shuffle_up(value, 16);
    if (lane_id >= 16) value = Op(value, temp);
    if (SCAN_TYPE == RADIX_SCAN_TYPE_EXCLUSIVE) {
        temp = simd_shuffle_up(value, 1);
        value = (lane_id == 0) ? 0 : temp;
    }
    return value;
}

/// Apply uniform value to thread-local array
template <ushort LENGTH, typename BinaryOp, typename T>
static inline void radix_thread_uniform_apply(threadgroup T* values, T uni, BinaryOp Op) {
    for (ushort i = 0; i < LENGTH; i++) {
        values[i] = Op(values[i], uni);
    }
}

// =============================================================================
// THREADGROUP PREFIX SCAN (RADIX-SPECIFIC)
// =============================================================================

/// Threadgroup prefix scan with sum output
template <int SCAN_TYPE, typename BinaryOp, typename T>
static T radix_threadgroup_prefix_scan_store_sum(
    T value,
    thread T& inclusive_sum,
    threadgroup T* shared,
    const ushort local_id,
    BinaryOp Op
) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_id < 32) {
        T partial_sum = radix_thread_scan<RADIX_BLOCK_SIZE / 32, SCAN_TYPE>(&shared[local_id * (RADIX_BLOCK_SIZE / 32)], Op);
        T prefix = radix_simdgroup_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(partial_sum, local_id, Op);
        radix_thread_uniform_apply<RADIX_BLOCK_SIZE / 32>(&shared[local_id * (RADIX_BLOCK_SIZE / 32)], prefix, Op);
        if (local_id == 31) shared[0] = prefix + partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (SCAN_TYPE == RADIX_SCAN_TYPE_INCLUSIVE) value = (local_id == 0) ? value : shared[local_id];
    else value = (local_id == 0) ? 0 : shared[local_id];
    inclusive_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

/// Threadgroup prefix scan without sum output
template <int SCAN_TYPE, typename BinaryOp, typename T>
static T radix_threadgroup_prefix_scan(T value, threadgroup T* shared, const ushort local_id, BinaryOp Op) {
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id < 32) {
        T partial_sum = radix_thread_scan<RADIX_BLOCK_SIZE / 32, SCAN_TYPE>(&shared[local_id * (RADIX_BLOCK_SIZE / 32)], Op);
        T prefix = radix_simdgroup_scan<RADIX_SCAN_TYPE_EXCLUSIVE>(partial_sum, local_id, Op);
        radix_thread_uniform_apply<RADIX_BLOCK_SIZE / 32>(&shared[local_id * (RADIX_BLOCK_SIZE / 32)], prefix, Op);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

// =============================================================================
// PARTIAL RADIX SORT (IN-THREADGROUP SORTING)
// =============================================================================

/// Sort by two bits (4-way partition)
template <typename T>
static T radix_sort_by_two_bits(const T value, threadgroup T* shared, const ushort local_id, const uchar current_bit) {
    uchar mask = value_to_key_at_bit<4>(value, current_bit);

    uchar4 partial_sum;
    uchar4 scan = {0};
    scan[mask] = 1;
    scan = radix_threadgroup_prefix_scan_store_sum<RADIX_SCAN_TYPE_EXCLUSIVE>(
        scan,
        partial_sum,
        reinterpret_cast<threadgroup uchar4*>(shared),
        local_id,
        radix_sum_op<uchar4>()
    );

    ushort4 offset;
    offset[0] = 0;
    offset[1] = offset[0] + partial_sum[0];
    offset[2] = offset[1] + partial_sum[1];
    offset[3] = offset[2] + partial_sum[2];

    shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T result = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return result;
}

/// Sort by one bit (2-way partition)
template <typename T>
static T radix_sort_by_bit(const T value, threadgroup T* shared, const ushort local_id, const uchar current_bit) {
    uchar mask = value_to_key_at_bit<2>(value, current_bit);

    uchar2 partial_sum;
    uchar2 scan = {0};
    scan[mask] = 1;
    scan = radix_threadgroup_prefix_scan_store_sum<RADIX_SCAN_TYPE_EXCLUSIVE>(
        scan,
        partial_sum,
        reinterpret_cast<threadgroup uchar2*>(shared),
        local_id,
        radix_sum_op<uchar2>()
    );

    ushort2 offset;
    offset[0] = 0;
    offset[1] = offset[0] + partial_sum[0];

    shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T result = shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return result;
}

/// Partial radix sort within threadgroup for current digit
template <typename T>
static T radix_partial_sort(const T value, threadgroup T* shared, const ushort local_id, const ushort current_digit) {
    T result = value;
    ushort current_bit = current_digit * radix_to_bits(RADIX_BUCKETS);
    const ushort key_bits = (ushort)(sizeof(RadixKeyType) * 8);
    const ushort range_end = (ushort)(current_bit + radix_to_bits(RADIX_BUCKETS));
    const ushort last_bit = min(range_end, key_bits);
    while (current_bit < last_bit) {
        ushort remaining = last_bit - current_bit;
        if (remaining >= 2) {
            result = radix_sort_by_two_bits(result, shared, local_id, current_bit);
            current_bit += 2;
        } else {
            result = radix_sort_by_bit(result, shared, local_id, current_bit);
            current_bit += 1;
        }
    }
    return result;
}

#endif // RADIX_SORT_HELPERS_H
