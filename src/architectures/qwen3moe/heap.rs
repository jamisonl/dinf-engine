// Adapted from the Apple MacOSX27.2 SDK libc++ heap algorithms.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt and heap-provenance.json for exact source hashes.
// This is the arithmetic-index branch used by the native custom-comparator sort.
// The caller checks finite scores and valid indices; equal scores keep libc++'s
// exact move order. Rust slice checks protect every access.

fn before(a: usize, b: usize, scores: &[f32]) -> bool {
    scores[a] > scores[b]
}

fn sift_up(indices: &mut [usize], scores: &[f32]) {
    let len = indices.len();
    if len <= 1 {
        return;
    }
    let mut parent = (len - 2) / 2;
    let mut last = len - 1;
    if before(indices[parent], indices[last], scores) {
        let value = indices[last];
        loop {
            indices[last] = indices[parent];
            last = parent;
            if parent == 0 {
                break;
            }
            parent = (parent - 1) / 2;
            if !before(indices[parent], value, scores) {
                break;
            }
        }
        indices[last] = value;
    }
}

// make_heap passes an odd prefix, so both children exist at each parent.
fn sift_down(indices: &mut [usize], mut start: usize, scores: &[f32]) {
    let len = indices.len();
    if len < 2 || start > (len - 2) / 2 {
        return;
    }
    let mut child = 2 * start + 1;
    child += usize::from(before(indices[child], indices[child + 1], scores));
    if before(indices[child], indices[start], scores) {
        return;
    }
    let top = indices[start];
    loop {
        indices[start] = indices[child];
        start = child;
        if child > (len - 2) / 2 {
            break;
        }
        child = 2 * child + 1;
        child += usize::from(before(indices[child], indices[child + 1], scores));
        if before(indices[child], top, scores) {
            break;
        }
    }
    indices[start] = top;
}

fn make_heap(indices: &mut [usize], scores: &[f32]) {
    let len = indices.len();
    if len <= 1 {
        return;
    }
    let sift_len = if len % 2 == 1 { len } else { len - 1 };
    // C++ calls a no-op sift_down at start 0 when sift_len is 1.
    if sift_len >= 3 {
        for start in (0..=(sift_len - 2) / 2).rev() {
            sift_down(&mut indices[..sift_len], start, scores);
        }
    }
    sift_up(indices, scores);
}

fn floyd_sift_down(indices: &mut [usize], scores: &[f32]) -> usize {
    let len = indices.len();
    let mut hole = 0;
    loop {
        let mut child = 2 * hole + 1;
        if child + 1 < len && before(indices[child], indices[child + 1], scores) {
            child += 1;
        }
        indices[hole] = indices[child];
        hole = child;
        if child > (len - 2) / 2 {
            return hole;
        }
    }
}

fn pop_heap(indices: &mut [usize], scores: &[f32]) {
    let len = indices.len();
    if len <= 1 {
        return;
    }
    let top = indices[0];
    let hole = floyd_sift_down(indices, scores);
    let last = len - 1;
    if hole == last {
        indices[hole] = top;
    } else {
        indices[hole] = indices[last];
        indices[last] = top;
        sift_up(&mut indices[..=hole], scores);
    }
}

/// Native libc++ heap fallback for a descending score comparator.
pub(super) fn heap_sort(indices: &mut [usize], scores: &[f32]) {
    assert!(indices.len() <= 128);
    make_heap(indices, scores);
    for len in (2..=indices.len()).rev() {
        pop_heap(&mut indices[..len], scores);
    }
}
