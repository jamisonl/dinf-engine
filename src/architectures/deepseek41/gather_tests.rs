//! Synthetic paged-gather tests: no model weights or inference.
use super::*;
use candle_core::Device;

fn payload(i: usize, bf16: bool) -> f32 {
    let bits = match i % 16 {
        0 => 0x8000_0000,
        1 => 0,
        2 => 0x7fc1_2345,
        3 => 0xffc5_4321,
        4 => 0x7f80_0000,
        5 => 0xff80_0000,
        _ => {
            ((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) & 0x807f_ffff)
                | (((i % 250 + 2) as u32) << 23)
        }
    };
    let value = f32::from_bits(bits);
    if bf16 && value.is_finite() {
        half::bf16::from_f32(value).to_f32()
    } else {
        value
    }
}

struct Fixture {
    chunks: Vec<Tensor>,
    host: Vec<f32>,
    ids: Tensor,
    selected: Vec<f32>,
    width: usize,
    rows: usize,
}
impl Fixture {
    fn new(sizes: &[usize], width: usize, d: &Device, bf16: bool) -> Result<Self> {
        let rows = sizes.iter().sum::<usize>();
        let host = (0..rows * width)
            .map(|i| payload(i, bf16))
            .collect::<Vec<_>>();
        let mut chunks = Vec::new();
        let mut base = 0;
        for &count in sizes {
            // Every source has a nonzero offset and poisoned future backing.
            let mut padded = vec![f32::from_bits(0x7fc0_dead); (count + 2) * width];
            padded[width..(count + 1) * width]
                .copy_from_slice(&host[base * width..(base + count) * width]);
            chunks.push(Tensor::from_vec(padded, (count + 2, width), d)?.narrow(0, 1, count)?);
            base += count;
        }
        let mut selected = (0..512)
            .map(|i| ((i * 7919 + rows - 1) % rows) as f32)
            .collect::<Vec<_>>();
        let mut edges = vec![0, 1.min(rows - 1), rows - 1, rows.saturating_sub(2)];
        for boundary in [
            255,
            256,
            257,
            23 * 256 - 1,
            23 * 256,
            24 * 256 - 1,
            24 * 256,
        ] {
            if boundary < rows {
                edges.push(boundary);
            }
        }
        let mut base = 0;
        for &count in sizes {
            base += count;
            edges.push(base - 1);
            if base < rows {
                edges.push(base);
            }
        }
        for (dst, &id) in selected.iter_mut().zip(edges.iter().cycle()).take(256) {
            *dst = id as f32;
        }
        // Duplicates, unsorted, and a nonzero-offset IDs view.
        selected[511] = selected[0];
        let mut padded = vec![-999.];
        padded.extend_from_slice(&selected);
        padded.push(-999.);
        let ids = Tensor::new(padded.as_slice(), d)?.narrow(0, 1, selected.len())?;
        Ok(Self {
            chunks,
            host,
            ids,
            selected,
            width,
            rows,
        })
    }
    fn expected(&self) -> Vec<u32> {
        self.selected
            .iter()
            .flat_map(|&id| {
                self.host[id as usize * self.width..(id as usize + 1) * self.width]
                    .iter()
                    .map(|v| v.to_bits())
            })
            .collect()
    }
    fn assert_inputs(&self) -> Result<()> {
        let actual = self
            .chunks
            .iter()
            .map(bits)
            .collect::<Result<Vec<_>>>()?
            .concat();
        assert_eq!(
            actual,
            self.host.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            bits(&self.ids)?,
            self.selected
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }
}
fn sizes(rows: usize) -> Vec<usize> {
    (0..rows)
        .step_by(256)
        .map(|base| (rows - base).min(256))
        .collect()
}
fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
fn deepseek41_gather_cpu_dispatch_bounds_and_geometry() -> Result<()> {
    for pages in [23usize, 24, 25] {
        for nonuniform in [false, true] {
            let mut geometry = (0..pages)
                .map(|i| if nonuniform { 1 + i * 13 % 319 } else { 256 })
                .collect::<Vec<_>>();
            *geometry.last_mut().unwrap() = 17;
            let f = Fixture::new(&geometry, 128, &Device::Cpu, false)?;
            assert_eq!(
                bits(&gather(&f.ids, &f.chunks, f.width, f.rows)?)?,
                f.expected()
            );
            f.assert_inputs()?;
        }
    }
    for rows in [1, 255, 256, 257, 511, 512, 513] {
        let f = Fixture::new(&sizes(rows), 128, &Device::Cpu, false)?;
        assert_eq!(
            bits(&gather(&f.ids, &f.chunks, f.width, rows)?)?,
            f.expected()
        );
        assert!(
            gather(
                &Tensor::new(&[rows as f32], &Device::Cpu)?,
                &f.chunks,
                128,
                rows
            )
            .is_err()
        );
        assert!(gather(&f.ids, &f.chunks, 128, rows + 1).is_err());
        assert!(gather(&f.ids, &f.chunks, 127, rows).is_err());
        assert!(gather(&f.ids, &f.chunks, usize::MAX, rows).is_err());
        assert!(gather(&f.ids, &f.chunks, 128, usize::MAX).is_err());
    }
    let ids = Tensor::new(&[0f32], &Device::Cpu)?;
    let good = Tensor::zeros((2, 128), DType::F32, &Device::Cpu)?;
    for bad in [
        Tensor::zeros((1, 128), DType::F64, &Device::Cpu)?,
        Tensor::zeros((128, 2), DType::F32, &Device::Cpu)?.t()?,
        Tensor::zeros(128, DType::F32, &Device::Cpu)?,
        Tensor::zeros((0, 128), DType::F32, &Device::Cpu)?,
    ] {
        assert!(gather(&ids, &[good.clone(), bad], 128, 3).is_err());
    }
    // Safe tensor construction rejects out-of-storage offsets; gather accepts
    // the final valid offset view, already exercised in every fixture above.
    assert!(good.narrow(0, 2, 1).is_err());
    Ok(())
}

#[cfg(target_os = "macos")]
fn output(f: &Fixture) -> Result<Tensor> {
    Tensor::zeros((512, f.width), DType::F32, f.ids.device())
}
#[cfg(target_os = "macos")]
fn encode(f: &Fixture, out: &Tensor, legacy: bool) -> Result<()> {
    super::metal::test_gather_into(&f.ids, &f.chunks, f.width, f.rows, out, legacy)
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_paged_gather_exact_long_prefixes() -> Result<()> {
    let d = Device::new_metal(0)?;
    for width in [128, 512] {
        for rows in [
            1,
            255,
            256,
            257,
            511,
            512,
            513,
            23 * 256 - 7,
            24 * 256,
            25 * 256 - 13,
            32426,
            64852,
        ] {
            let f = Fixture::new(&sizes(rows), width, &d, false)?;
            let reference = output(&f)?;
            encode(&f, &reference, true)?;
            let candidate = gather(&f.ids, &f.chunks, width, rows)?;
            let cpu_chunks = f
                .chunks
                .iter()
                .map(|t| t.to_device(&Device::Cpu))
                .collect::<Result<Vec<_>>>()?;
            let oracle = gather(&f.ids.to_device(&Device::Cpu)?, &cpu_chunks, width, rows)?;
            assert_eq!(
                bits(&reference)?,
                f.expected(),
                "old GPU width={width} rows={rows}"
            );
            assert_eq!(
                bits(&candidate)?,
                bits(&oracle)?,
                "new GPU width={width} rows={rows}"
            );
            assert_eq!(bits(&candidate)?, f.expected());
            f.assert_inputs()?;
        }
        let geometry = (0..25).map(|i| 1 + i * 17 % 319).collect::<Vec<_>>();
        let f = Fixture::new(&geometry, width, &d, true)?;
        let old = output(&f)?;
        encode(&f, &old, true)?;
        assert_eq!(bits(&old)?, f.expected());
        assert_eq!(
            bits(&gather(&f.ids, &f.chunks, width, f.rows)?)?,
            f.expected()
        );
        f.assert_inputs()?;
    }
    // Guard invalid IDs without touching poisoned future rows. Sentinel output
    // makes missing writes observable; legacy comparison above uses valid IDs.
    let f = Fixture::new(&sizes(24 * 256 + 17), 512, &d, false)?;
    let sentinel = Tensor::full(-123f32, (7, 512), &d)?;
    let invalid = Tensor::new(
        &[
            -1.,
            f.rows as f32,
            (f.rows + 1) as f32,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
        ],
        &d,
    )?;
    super::metal::test_gather_into(&invalid, &f.chunks, 512, f.rows, &sentinel, false)?;
    assert_eq!(bits(&sentinel)?, vec![(-123f32).to_bits(); 7 * 512]);
    // Last-page validation error must not disturb an already queued gather.
    let pending = gather(&f.ids, &f.chunks, 512, f.rows)?;
    let mut bad = f.chunks.clone();
    bad.push(Tensor::zeros((1, 512), DType::F32, &Device::Cpu)?);
    assert!(gather(&f.ids, &bad, 512, f.rows + 1).is_err());
    d.synchronize()?;
    assert_eq!(bits(&pending)?, f.expected());
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_paged_gather_singleton_bounds_and_lifetime() -> Result<()> {
    let d = Device::new_metal(0)?;
    for width in [1, 129, 513] {
        for pages in [1usize, 25] {
            let mut geometry = vec![256; pages];
            *geometry.last_mut().unwrap() = 17;
            let mut f = Fixture::new(&geometry, width, &d, false)?;
            f.chunks = f
                .chunks
                .iter()
                .map(crate::architectures::copy_persistent_tensor)
                .collect::<anyhow::Result<Vec<_>>>()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let expected = f.expected();
            // Singleton-only and final batches, odd widths/tails, offset IDs,
            // NaN payloads and signed zeros must survive source handle drops.
            let pending = gather(&f.ids, &f.chunks, width, f.rows)?;
            let invalid = Tensor::new(
                &[
                    -1.,
                    f.rows as f32,
                    (f.rows + 1) as f32,
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::MAX,
                ],
                &d,
            )?;
            let sentinel = Tensor::full(-123f32, (7, width), &d)?;
            super::metal::test_gather_into(&invalid, &f.chunks, width, f.rows, &sentinel, false)?;
            drop((f, invalid));
            let pressure = Tensor::full(-777f32, (257, width), &d)?;
            d.synchronize()?;
            assert_eq!(bits(&pending)?, expected, "pages={pages} width={width}");
            assert_eq!(bits(&sentinel)?, vec![(-123f32).to_bits(); 7 * width]);
            drop(pressure);
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_paged_gather_completion_and_fork_lifetime() -> Result<()> {
    if crate::test_in_subprocess(concat!(
        module_path!(),
        "::deepseek41_metal_paged_gather_completion_and_fork_lifetime"
    )) {
        return Ok(());
    }
    let d = Device::new_metal(0)?;
    let md = d.as_metal_device()?;
    // Warm pipelines and pool buckets before measuring private source retention.
    let warm = Fixture::new(&sizes(25 * 256 + 17), 512, &d, false)?;
    let warm_out = gather(&warm.ids, &warm.chunks, 512, warm.rows)?;
    d.synchronize()?;
    drop((warm, warm_out));
    let baseline = md.metal_device().current_allocated_size();
    let mut samples = Vec::new();
    for _ in 0..2 {
        let mut f = Fixture::new(&sizes(25 * 256 + 17), 512, &d, false)?;
        // Existing exact private-copy route avoids pool retention hiding leaks.
        f.chunks = f
            .chunks
            .iter()
            .map(crate::architectures::copy_persistent_tensor)
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let expected = f.expected();
        let first = gather(&f.ids, &f.chunks, 512, f.rows)?;
        let mut fork = f.chunks.clone();
        // Immutable fork replaces only the tail; the queued first gather must
        // retain the old tail after every original source handle is dropped.
        let tail = *fork.last().unwrap().dims().first().unwrap();
        fork.pop();
        fork.push(Tensor::full(19f32, (tail, 512), &d)?);
        let second = gather(&f.ids, &fork, 512, f.rows)?;
        let mut expected_fork = expected.clone();
        for (row, &id) in f.selected.iter().enumerate() {
            if id as usize >= f.rows - tail {
                expected_fork[row * 512..(row + 1) * 512].fill(19f32.to_bits());
            }
        }
        drop((f, fork));
        let pressure = Tensor::full(-777f32, (256, 512), &d)?;
        d.synchronize()?; // Existing completion boundary, not a production fence.
        assert_eq!(bits(&first)?, expected);
        assert_eq!(bits(&second)?, expected_fork);
        drop((first, second, pressure));
        d.synchronize()?;
        samples.push(md.metal_device().current_allocated_size());
    }
    // Pool output/pressure buckets can remain. Private source sets (13 MiB each)
    // must not accumulate across the two completed iterations.
    assert!(
        samples[1] <= samples[0] + 1024 * 1024,
        "retention grew: {samples:?}"
    );
    assert!(
        samples[1] <= baseline + 4 * 1024 * 1024,
        "private sources retained: baseline={baseline} samples={samples:?}"
    );
    println!("gather lifetime allocated bytes baseline={baseline} after={samples:?}");
    let evidence = serde_json::json!({"baseline_device_allocated_bytes":baseline,
        "after_completed_scope_device_allocated_bytes":samples,
        "tracking":"existing Metal current_allocated_size; warmed pool; private exact-copy sources",
        "fork_outputs_bit_exact":true,"dropped_sources_survive_completion":true});
    println!("gather lifetime evidence: {evidence}");
    Ok(())
}
