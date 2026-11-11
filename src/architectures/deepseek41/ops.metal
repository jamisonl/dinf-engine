// Maya-XS cache primitives, derived from ds4.c metal/argsort.metal.
// See adjacent NOTICE and LICENSE-MIT.
#include <metal_stdlib>
using namespace metal;
struct Args { uint n, keep, run, stride, width, base, count, wide; };

// One SIMD group per (query,key), one dispatch per immutable page for the
// ENTIRE tile. Heads are reduced internally; only [T,Nmax] leaves the kernel.
kernel void maya_index_many(
    device const float *q [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *keys [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant Args &a [[buffer(4)]],
    constant uint *visible [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.y, key = a.base + group.x;
    if (group.x >= a.count || row >= a.n) return;
    if (key >= visible[row]) {
        if (lane == 0) out[ulong(row) * a.width + key] = -INFINITY;
        return; // In particular, no future poisoned key load.
    }
    float score = 0.0f;
    for (uint head = 0; head < 32; ++head) {
        float dot = 0.0f;
        for (uint c = lane; c < 128; c += 32)
            dot += q[(ulong(row) * 32 + head) * 128 + c]
                * keys[ulong(group.x) * 128 + c];
        dot = simd_sum(dot);
        if (lane == 0) score += max(dot, 0.0f) * weights[row * 32 + head];
    }
    if (lane == 0) out[ulong(row) * a.width + key] = score * (1.0f / 64.0f);
}

kernel void maya_embedding(device const uchar *packed [[buffer(0)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.width) return;
    ulong block = (ulong(a.base) * (a.width / 32u) + i / 32u) * 34u;
    ushort bits = ushort(packed[block]) | (ushort(packed[block + 1]) << 8);
    out[i] = float(as_type<half>(bits)) * float(as_type<char>(packed[block + 2 + i % 32]));
}
kernel void maya_blocks(device const float *s [[buffer(0)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint blocks = (a.n + 7) / 8;
    if (i >= blocks) return;
    float best = -INFINITY;
    for (uint j = i * 8; j < min(a.n, (i + 1) * 8); ++j) best = max(best, s[j]);
    out[i] = i == blocks - 1 ? INFINITY : best;
}
kernel void maya_mask(device const float *ids [[buffer(0)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.count) out[i] = -INFINITY;
}
kernel void maya_mask_scatter(device const float *ids [[buffer(0)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) out[uint(ids[i])] = 0.0f;
}
kernel void maya_filter(device const float *s [[buffer(0)]], device const float *mask [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) out[i] = mask[i / 8] == 0.0f ? s[i] : -INFINITY;
}
// Singleton batches bind one source, with the same validated causal frontier.
kernel void maya_gather_page(device const float *ids [[buffer(0)]],
    device const uint *page [[buffer(1)]], device uint *out [[buffer(2)]],
    constant Args &a [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= a.n) return;
    float selected = ids[row];
    if (!isfinite(selected) || selected < 0.0f || selected >= float(a.keep)) return;
    uint id = uint(selected);
    if (id >= a.keep || id < a.base || id - a.base >= a.count) return;
    for (ulong i = lane; i < a.width; i += 256)
        out[ulong(row) * a.width + i] = page[ulong(id - a.base) * a.width + i];
}
struct Span { uint base, count; };
kernel void maya_gather_pages(
    device const float *ids [[buffer(0)]],
    constant Span *spans [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant Args &a [[buffer(3)]],
    device const uint *p0 [[buffer(4)]],
    device const uint *p1 [[buffer(5)]],
    device const uint *p2 [[buffer(6)]],
    device const uint *p3 [[buffer(7)]],
    device const uint *p4 [[buffer(8)]],
    device const uint *p5 [[buffer(9)]],
    device const uint *p6 [[buffer(10)]],
    device const uint *p7 [[buffer(11)]],
    device const uint *p8 [[buffer(12)]],
    device const uint *p9 [[buffer(13)]],
    device const uint *p10 [[buffer(14)]],
    device const uint *p11 [[buffer(15)]],
    device const uint *p12 [[buffer(16)]],
    device const uint *p13 [[buffer(17)]],
    device const uint *p14 [[buffer(18)]],
    device const uint *p15 [[buffer(19)]],
    device const uint *p16 [[buffer(20)]],
    device const uint *p17 [[buffer(21)]],
    device const uint *p18 [[buffer(22)]],
    device const uint *p19 [[buffer(23)]],
    device const uint *p20 [[buffer(24)]],
    device const uint *p21 [[buffer(25)]],
    device const uint *p22 [[buffer(26)]],
    device const uint *p23 [[buffer(27)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    if (row >= a.n) return;
    float selected = ids[row];
    // Check before float->uint conversion and before any page access.
    if (!isfinite(selected) || selected < 0.0f || selected >= float(a.keep)) return;
    uint id = uint(selected);
    if (id >= a.keep) return;
    for (uint page = 0; page < a.count; ++page) {
        Span span = spans[page];
        if (id < span.base || id - span.base >= span.count) continue;
        device const uint *src;
        switch (page) {
            case 0: src = p0; break;
            case 1: src = p1; break;
            case 2: src = p2; break;
            case 3: src = p3; break;
            case 4: src = p4; break;
            case 5: src = p5; break;
            case 6: src = p6; break;
            case 7: src = p7; break;
            case 8: src = p8; break;
            case 9: src = p9; break;
            case 10: src = p10; break;
            case 11: src = p11; break;
            case 12: src = p12; break;
            case 13: src = p13; break;
            case 14: src = p14; break;
            case 15: src = p15; break;
            case 16: src = p16; break;
            case 17: src = p17; break;
            case 18: src = p18; break;
            case 19: src = p19; break;
            case 20: src = p20; break;
            case 21: src = p21; break;
            case 22: src = p22; break;
            case 23: src = p23; break;
            default: return;
        }
        // Integer loads/stores preserve NaN payloads and signed zero exactly.
        for (ulong i = lane; i < a.width; i += 256)
            out[ulong(row) * a.width + i] = src[ulong(id - span.base) * a.width + i];
        return;
    }
}
inline bool better(uint x, uint y, device const float *scores, constant Args &a) {
    return x < a.n && (y >= a.n || scores[x] > scores[y] || (a.wide && scores[x] == scores[y] && x < y));
}
kernel void maya_sort_runs(device const float *scores [[buffer(0)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint ids[1024];
    uint base = group * a.run;
    ids[tid] = base + tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= a.run; k *= 2) {
        for (uint j = k / 2; j; j /= 2) {
            uint other = tid ^ j;
            if (other > tid) {
                uint x = ids[tid], y = ids[other];
                bool swap = (tid & k) == 0 ? (x >= a.n || better(y, x, scores, a)) : (y >= a.n || better(x, y, scores, a));
                if (swap) { ids[tid] = y; ids[other] = x; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < a.stride) out[group * a.stride + tid] = ids[tid] < a.n ? float(ids[tid]) : -1.0f;
}
kernel void maya_merge(device const float *scores [[buffer(0)]], device const float *runs [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    uint groups = (a.n + 2 * a.run - 1) / (2 * a.run);
    if (t >= groups * a.count) return;
    uint group = t / a.count, rank = t % a.count;
    uint base = group * 2 * a.run;
    uint left_len = min(a.stride, a.n - base);
    uint right_len = base + a.run < a.n ? min(a.stride, a.n - base - a.run) : 0;
    if (rank >= left_len + right_len) { out[t] = -1.0f; return; }
    uint left_base = group * 2 * a.stride, right_base = left_base + a.stride;
    uint lo = rank > right_len ? rank - right_len : 0, hi = min(rank, left_len);
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        uint x = uint(runs[left_base + mid]), y = uint(runs[right_base + rank - mid - 1]);
        bool left = scores[x] > scores[y] || (scores[x] == scores[y] && (!a.wide || x < y));
        if (left) lo = mid + 1; else hi = mid;
    }
    uint i = lo, j = rank - i;
    if (i == left_len) out[t] = runs[right_base + j];
    else if (j == right_len) out[t] = runs[left_base + i];
    else {
        uint x = uint(runs[left_base + i]), y = uint(runs[right_base + j]);
        bool left = scores[x] > scores[y] || (scores[x] == scores[y] && (!a.wide || x < y));
        out[t] = float(left ? x : y);
    }
}
