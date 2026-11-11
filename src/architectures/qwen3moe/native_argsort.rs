// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Adapted from the LLVM Project's libc++ __algorithm/sort.h.
// Original notice: Part of the LLVM Project, under the Apache License v2.0
// with LLVM Exceptions. See LICENSE-LLVM and https://llvm.org/LICENSE.txt.
//
// Modified for dinf: safe Rust, finite F32 scores, integer indices, descending
// custom comparator, and at most 128 entries. This preserves the pinned native
// CPU router's full sort order, including equal scores. It is not a new general
// tie-breaking rule. Header SHA256:
// b644503828e13c3c90b5d343684eb993add9330762c5a7957001876be653e014.

#[path = "heap.rs"]
mod native_heap;

/// The pinned libc++ custom-comparator order, including ties.
pub(super) fn descending_indices(scores: &[f32]) -> Result<Vec<usize>, &'static str> {
    if scores.is_empty() || scores.len() > 128 || scores.iter().any(|x| !x.is_finite()) {
        return Err("native router sort needs 1..128 finite scores");
    }
    let mut ids: Vec<_> = (0..scores.len()).collect();
    let depth = 2 * scores.len().ilog2() as usize;
    let end = ids.len();
    introsort(&mut ids, scores, 0, end, depth, true);
    Ok(ids)
}

#[inline]
fn before(scores: &[f32], a: usize, b: usize) -> bool {
    scores[a] > scores[b]
}

fn sort3(ids: &mut [usize], scores: &[f32], x: usize, y: usize, z: usize) {
    if !before(scores, ids[y], ids[x]) {
        if !before(scores, ids[z], ids[y]) {
            return;
        }
        ids.swap(y, z);
        if before(scores, ids[y], ids[x]) {
            ids.swap(x, y);
        }
        return;
    }
    if before(scores, ids[z], ids[y]) {
        ids.swap(x, z);
        return;
    }
    ids.swap(x, y);
    if before(scores, ids[z], ids[y]) {
        ids.swap(y, z);
    }
}

// The guarded and unguarded native forms have the same moves: the native
// unguarded form relies on a preceding partition sentinel. Keep a bound here.
fn insertion(ids: &mut [usize], scores: &[f32], first: usize, last: usize) {
    for i in first.saturating_add(1)..last {
        let value = ids[i];
        let mut j = i;
        while j > first && before(scores, value, ids[j - 1]) {
            ids[j] = ids[j - 1];
            j -= 1;
        }
        ids[j] = value;
    }
}

fn insertion_incomplete(ids: &mut [usize], scores: &[f32], first: usize, last: usize) -> bool {
    if last - first <= 5 {
        // libc++'s small custom-comparator sorts are stable.
        insertion(ids, scores, first, last);
        return true;
    }
    sort3(ids, scores, first, first + 1, first + 2);
    let mut count = 0;
    for i in first + 3..last {
        if before(scores, ids[i], ids[i - 1]) {
            let value = ids[i];
            let mut j = i;
            while j > first && before(scores, value, ids[j - 1]) {
                ids[j] = ids[j - 1];
                j -= 1;
            }
            ids[j] = value;
            count += 1;
            if count == 8 {
                return i + 1 == last;
            }
        }
    }
    true
}

fn partition_right(ids: &mut [usize], scores: &[f32], begin: usize, end: usize) -> (usize, bool) {
    let pivot = ids[begin];
    let mut first = begin + 1;
    while before(scores, ids[first], pivot) {
        first += 1;
    }
    let mut last = end;
    if begin == first - 1 {
        while first < last {
            last -= 1;
            if before(scores, ids[last], pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if before(scores, ids[last], pivot) {
                break;
            }
        }
    }
    let already_partitioned = first >= last;
    while first < last {
        ids.swap(first, last);
        loop {
            first += 1;
            if !before(scores, ids[first], pivot) {
                break;
            }
        }
        loop {
            last -= 1;
            if before(scores, ids[last], pivot) {
                break;
            }
        }
    }
    let position = first - 1;
    if begin != position {
        ids[begin] = ids[position];
    }
    ids[position] = pivot;
    (position, already_partitioned)
}

fn partition_left(ids: &mut [usize], scores: &[f32], begin: usize, end: usize) -> usize {
    let pivot = ids[begin];
    let mut first = begin;
    let mut last = end;
    if before(scores, pivot, ids[end - 1]) {
        loop {
            first += 1;
            if before(scores, pivot, ids[first]) {
                break;
            }
        }
    } else {
        first += 1;
        while first < last && !before(scores, pivot, ids[first]) {
            first += 1;
        }
    }
    if first < last {
        loop {
            last -= 1;
            if !before(scores, pivot, ids[last]) {
                break;
            }
        }
    }
    while first < last {
        ids.swap(first, last);
        loop {
            first += 1;
            if before(scores, pivot, ids[first]) {
                break;
            }
        }
        loop {
            last -= 1;
            if !before(scores, pivot, ids[last]) {
                break;
            }
        }
    }
    let position = first - 1;
    if begin != position {
        ids[begin] = ids[position];
    }
    ids[position] = pivot;
    first
}

fn introsort(
    ids: &mut [usize],
    scores: &[f32],
    mut first: usize,
    mut last: usize,
    mut depth: usize,
    mut leftmost: bool,
) {
    loop {
        let len = last - first;
        if len < 24 {
            insertion(ids, scores, first, last);
            return;
        }
        if depth == 0 {
            native_heap::heap_sort(&mut ids[first..last], scores);
            return;
        }
        depth -= 1;
        // The native ninther branch is only used above128, excluded by the API.
        sort3(ids, scores, first + len / 2, first, last - 1);
        if !leftmost && !before(scores, ids[first - 1], ids[first]) {
            first = partition_left(ids, scores, first, last);
            continue;
        }
        let (pivot, already_partitioned) = partition_right(ids, scores, first, last);
        if already_partitioned {
            let left_sorted = insertion_incomplete(ids, scores, first, pivot);
            if insertion_incomplete(ids, scores, pivot + 1, last) {
                if left_sorted {
                    return;
                }
                last = pivot;
                continue;
            }
            if left_sorted {
                first = pivot + 1;
                continue;
            }
        }
        introsort(ids, scores, first, pivot, depth, leftmost);
        leftmost = false;
        first = pivot + 1;
    }
}
