//! BF16 text projections through the public MPSGraph matrix operation.
//! Inputs, weights, and outputs stay BF16. Graphs retain shapes, never model data.
//! Built only on macOS ARM64 (platform minimum macOS11). This path requires
//! macOS15 and an MPS-capable Apple7 device; other devices use Candle.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use objc2::rc::{Allocated, Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2::{AnyThread, extern_class, extern_conformance, extern_methods};
use objc2_foundation::{
    CopyingHelper, NSArray, NSCopying, NSDictionary, NSNumber, NSObject, NSOperatingSystemVersion,
    NSProcessInfo, NSString,
};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice, MTLGPUFamily};
use std::{cell::RefCell, collections::HashMap};

#[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
unsafe extern "C" {}
#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {
    fn MPSSupportsMTLDevice(device: &ProtocolObject<dyn MTLDevice>) -> objc2::runtime::Bool;
}

// The declarations below match the macOS SDK's public MPSGraph and MPSNDArray
// headers. NDArray's buffer-offset initializer requires macOS 15 or newer.
extern_class!(
    #[unsafe(super(NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct MPSGraphObject;
);
extern_class!(
    #[unsafe(super(MPSGraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct MPSGraph;
);
extern_class!(
    #[unsafe(super(MPSGraphObject, NSObject))]
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct MPSGraphTensor;
);
extern_class!(
    #[unsafe(super(NSObject))]
    #[derive(Debug)]
    struct MPSGraphTensorData;
);
extern_class!(
    #[unsafe(super(NSObject))]
    #[derive(Debug)]
    struct MPSNDArrayDescriptor;
);
extern_class!(
    #[unsafe(super(NSObject))]
    #[derive(Debug)]
    struct MPSNDArray;
);
extern_conformance!(
    unsafe impl NSCopying for MPSGraphTensor {}
);
unsafe impl CopyingHelper for MPSGraphTensor {
    type Result = Self;
}

impl MPSGraph {
    extern_methods!(
        #[unsafe(method(new))]
        fn new() -> Retained<Self>;
        #[unsafe(method(placeholderWithShape:dataType:name:))]
        unsafe fn placeholder(
            &self,
            shape: &NSArray<NSNumber>,
            dtype: u32,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;
        #[unsafe(method(transposeTensor:dimension:withDimension:name:))]
        unsafe fn transpose(
            &self,
            tensor: &MPSGraphTensor,
            axis: usize,
            other: usize,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;
        #[unsafe(method(matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:))]
        unsafe fn matmul(
            &self,
            a: &MPSGraphTensor,
            b: &MPSGraphTensor,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;
        #[unsafe(method(castTensor:toType:name:))]
        unsafe fn cast(
            &self,
            tensor: &MPSGraphTensor,
            dtype: u32,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;
        #[unsafe(method(softMaxWithTensor:axis:name:))]
        unsafe fn softmax(
            &self,
            tensor: &MPSGraphTensor,
            axis: isize,
            name: Option<&NSString>,
        ) -> Retained<MPSGraphTensor>;
        #[unsafe(method(runWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:))]
        unsafe fn run(
            &self,
            queue: &ProtocolObject<dyn MTLCommandQueue>,
            feeds: &NSDictionary<MPSGraphTensor, MPSGraphTensorData>,
            operations: Option<&NSArray<NSObject>>,
            results: &NSDictionary<MPSGraphTensor, MPSGraphTensorData>,
        );
    );
}
impl MPSNDArrayDescriptor {
    extern_methods!(
        #[unsafe(method(descriptorWithDataType:shape:))]
        unsafe fn descriptor(dtype: u32, shape: &NSArray<NSNumber>) -> Retained<Self>;
        #[unsafe(method(setPreferPackedRows:))]
        unsafe fn set_packed_rows(&self, packed: bool);
    );
}
impl MPSNDArray {
    extern_methods!(
        #[unsafe(method(initWithBuffer:offset:descriptor:))]
        unsafe fn with_buffer(
            this: Allocated<Self>,
            buffer: &ProtocolObject<dyn MTLBuffer>,
            offset: usize,
            descriptor: &MPSNDArrayDescriptor,
        ) -> Retained<Self>;
    );
}
impl MPSGraphTensorData {
    extern_methods!(
        #[unsafe(method(initWithMPSNDArray:))]
        unsafe fn with_array(this: Allocated<Self>, array: &MPSNDArray) -> Retained<Self>;
    );
}

const BF16: u32 = 0x9000_0010;
const MAX_WEIGHT_BYTES: usize = 256 * 1024 * 1024;
const MAX_ACTIVATION_BYTES: usize = 64 * 1024 * 1024;
const MAX_GRAPHS: usize = 16;
type Key = (DeviceId, usize, usize, usize);
struct Graph {
    graph: Retained<MPSGraph>,
    a: Retained<MPSGraphTensor>,
    b: Retained<MPSGraphTensor>,
    c: Retained<MPSGraphTensor>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}
thread_local! { static GRAPHS: RefCell<HashMap<Key, Graph>> = RefCell::new(HashMap::new()); }
fn shape(rows: usize, cols: usize) -> Retained<NSArray<NSNumber>> {
    NSArray::from_retained_slice(&[NSNumber::new_usize(rows), NSNumber::new_usize(cols)])
}
fn supported_platform() -> bool {
    NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
        majorVersion: 15,
        minorVersion: 0,
        patchVersion: 0,
    })
}
// Keep this path on public Apple GPU/MPS support. The packed buffer initializer
// requires macOS 15; other devices use the existing Candle path.
pub(super) fn supported_device(device: &candle_core::Device) -> bool {
    if !supported_platform() {
        return false;
    }
    let Ok(device) = device.as_metal_device() else {
        return false;
    };
    let metal = device.metal_device().as_ref();
    metal.supportsFamily(MTLGPUFamily::Apple7) && unsafe { MPSSupportsMTLDevice(metal).as_bool() }
}
fn byte_count(rows: usize, cols: usize, cap: usize) -> Option<usize> {
    rows.checked_mul(cols)?
        .checked_mul(2)
        .filter(|n| *n > 0 && *n <= cap)
}
pub(super) fn try_linear(input: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    if !input.device().is_metal()
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.device().same_device(weight.device())
        || !matches!(input.rank(), 2 | 3)
        || weight.rank() != 2
        || !supported_device(input.device())
    {
        return Ok(None);
    }
    let (columns, inner) = weight.dims2()?;
    if inner == 0 || input.dims().last() != Some(&inner) {
        return Ok(None);
    }
    let rows = input.elem_count() / inner;
    if byte_count(columns, inner, MAX_WEIGHT_BYTES).is_none()
        || byte_count(rows, inner, MAX_ACTIVATION_BYTES).is_none()
        || byte_count(rows, columns, MAX_ACTIVATION_BYTES).is_none()
    {
        return Ok(None);
    }
    Ok(Some(
        input
            .contiguous()?
            .apply_op2_no_bwd(&weight.contiguous()?, &Dense)?,
    ))
}
fn offset(storage: &MetalStorage, layout: &Layout, cap: usize) -> Result<usize> {
    let bytes = layout
        .shape()
        .elem_count()
        .checked_mul(2)
        .filter(|n| *n > 0 && *n <= cap)
        .ok_or_else(|| candle_core::Error::Msg("MPS text buffer count".into()))?;
    let offset = layout
        .start_offset()
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg("MPS text offset".into()))?;
    if storage.dtype() != DType::BF16
        || !layout.is_contiguous()
        || offset
            .checked_add(bytes)
            .is_none_or(|end| end > storage.buffer().length())
    {
        candle_core::bail!("MPS text buffer span")
    }
    Ok(offset)
}
// Callers check the exact BF16 byte span and contiguous shape before this SDK
// initializer. Retained tensor data owns its NDArray for the synchronous run.
unsafe fn data(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Retained<MPSGraphTensorData> {
    unsafe { data_shape(buffer, offset, &[rows, cols]) }
}
unsafe fn data_shape(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    dims: &[usize],
) -> Retained<MPSGraphTensorData> {
    let dimensions = NSArray::from_retained_slice(
        &dims
            .iter()
            .map(|&d| NSNumber::new_usize(d))
            .collect::<Vec<_>>(),
    );
    let desc = unsafe { MPSNDArrayDescriptor::descriptor(BF16, &dimensions) };
    unsafe { desc.set_packed_rows(true) };
    let array = unsafe { MPSNDArray::with_buffer(MPSNDArray::alloc(), buffer, offset, &desc) };
    unsafe { MPSGraphTensorData::with_array(MPSGraphTensorData::alloc(), &array) }
}
struct Dense;
impl CustomOp2 for Dense {
    fn name(&self) -> &'static str {
        "text-bf16-mpsgraph"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("MPS text requires Metal")
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        il: &Layout,
        weight: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let io = offset(input, il, MAX_ACTIVATION_BYTES)?;
        let wo = offset(weight, wl, MAX_WEIGHT_BYTES)?;
        let (columns, inner) = wl.shape().dims2()?;
        if input.device().id() != weight.device().id()
            || inner == 0
            || il.dims().last() != Some(&inner)
        {
            candle_core::bail!("MPS text shape/device")
        }
        let rows = il.shape().elem_count() / inner;
        let bytes = byte_count(rows, columns, MAX_ACTIVATION_BYTES)
            .ok_or_else(|| candle_core::Error::Msg("MPS text output bound".into()))?;
        let device = input.device();
        // Finish prior Candle writes before the separate MPS queue reads them.
        // MPSGraph's run method blocks through completion before returning.
        device.wait_until_completed()?;
        let output = device
            .new_buffer_builder()
            .with_size_for(bytes / 2, DType::BF16)
            .with_label(self.name())
            .build()?;
        autoreleasepool(|_| -> Result<()> {
            GRAPHS.with(|graphs| -> Result<()> {
                let mut graphs = graphs.borrow_mut();
                let key = (device.id(), rows, columns, inner);
                if !graphs.contains_key(&key) {
                    if graphs.len() >= MAX_GRAPHS {
                        graphs.clear();
                    }
                    let queue = device
                        .metal_device()
                        .as_ref()
                        .newCommandQueue()
                        .ok_or_else(|| candle_core::Error::Msg("MPS text queue".into()))?;
                    let graph = MPSGraph::new();
                    // All shapes and data types have passed the byte caps above.
                    let a = unsafe { graph.placeholder(&shape(rows, inner), BF16, None) };
                    let b = unsafe { graph.placeholder(&shape(columns, inner), BF16, None) };
                    let transpose = unsafe { graph.transpose(&b, 0, 1, None) };
                    let c = unsafe { graph.matmul(&a, &transpose, None) };
                    graphs.insert(
                        key,
                        Graph {
                            graph,
                            a,
                            b,
                            c,
                            queue,
                        },
                    );
                }
                let g = &graphs[&key];
                let a = unsafe { data(input.buffer().as_ref(), io, rows, inner) };
                let b = unsafe { data(weight.buffer().as_ref(), wo, columns, inner) };
                let c = unsafe { data(output.as_ref().as_ref(), 0, rows, columns) };
                let feeds = NSDictionary::from_slices(&[&*g.a, &*g.b], &[&*a, &*b]);
                let results = NSDictionary::from_slices(&[&*g.c], &[&*c]);
                // The synchronous call keeps every source/output buffer alive.
                unsafe {
                    g.graph.run(&g.queue, &feeds, None, &results);
                }
                Ok(())
            })
        })?;
        let mut shape = il.dims().to_vec();
        *shape
            .last_mut()
            .ok_or_else(|| candle_core::Error::Msg("MPS text rank".into()))? = columns;
        Ok((
            MetalStorage::new(output, device.clone(), bytes / 2, DType::BF16),
            Shape::from(shape),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use half::bf16;

    #[test]
    fn bounds_and_cpu_decline() -> Result<()> {
        assert_eq!(byte_count(3, 9, 54), Some(54));
        assert_eq!(byte_count(3, 9, 53), None);
        assert_eq!(byte_count(usize::MAX, 2, usize::MAX), None);
        assert_eq!(byte_count(0, 2, usize::MAX), None);
        let x = Tensor::zeros((1, 3, 32), DType::BF16, &Device::Cpu)?;
        let w = Tensor::zeros((9, 32), DType::BF16, &Device::Cpu)?;
        assert!(try_linear(&x, &w)?.is_none());
        Ok(())
    }

    #[test]
    fn packed_rows_views_and_f64() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supported_device(&device) {
            return Ok(());
        }
        let x = (0..3 * 32)
            .map(|i| bf16::from_f32((i % 13) as f32 / 16.0))
            .collect::<Vec<_>>();
        let w = (0..9 * 32)
            .map(|i| bf16::from_f32((i % 7) as f32 / 32.0))
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for row in 0..3 {
            for col in 0..9 {
                let dot = (0..32)
                    .map(|k| {
                        f64::from(x[row * 32 + k].to_f32()) * f64::from(w[col * 32 + k].to_f32())
                    })
                    .sum::<f64>();
                expected.push(bf16::from_f64(dot).to_bits());
            }
        }
        for offset in [0, 7] {
            let mut xp = vec![bf16::ZERO; offset];
            xp.extend_from_slice(&x);
            let mut wp = vec![bf16::ZERO; offset];
            wp.extend_from_slice(&w);
            let xl = xp.len();
            let wl = wp.len();
            let tx = Tensor::from_vec(xp, xl, &device)?
                .narrow(0, offset, x.len())?
                .reshape((1, 3, 32))?;
            let tw = Tensor::from_vec(wp, wl, &device)?
                .narrow(0, offset, w.len())?
                .reshape((9, 32))?;
            let actual = try_linear(&tx, &tw)?
                .unwrap()
                .flatten_all()?
                .to_vec1::<bf16>()?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            assert!(try_linear(&tx.to_dtype(DType::F32)?, &tw.to_dtype(DType::F32)?)?.is_none());
            assert!(try_linear(&tx, &tw.narrow(1, 0, 31)?)?.is_none());
        }
        let ix = x.iter().flat_map(|v| [*v, bf16::ONE]).collect::<Vec<_>>();
        let iw = w.iter().flat_map(|v| [*v, bf16::ONE]).collect::<Vec<_>>();
        let tx = Tensor::from_vec(ix, (3, 32, 2), &device)?
            .narrow(2, 0, 1)?
            .squeeze(2)?;
        let tw = Tensor::from_vec(iw, (9, 32, 2), &device)?
            .narrow(2, 0, 1)?
            .squeeze(2)?;
        let actual = try_linear(&tx, &tw)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<bf16>()?
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }
}

// Attention uses batched matrix operations. Each run returns a BF16 buffer,
// keeping the reference's rounding before score scaling, masking and softmax.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum MathOp {
    Matmul(bool),
    Softmax,
}
type MathKey = (DeviceId, MathOp, Vec<usize>, Vec<usize>);
struct MathGraph {
    graph: Retained<MPSGraph>,
    a: Retained<MPSGraphTensor>,
    b: Option<Retained<MPSGraphTensor>>,
    c: Retained<MPSGraphTensor>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}
thread_local! { static MATH_GRAPHS: RefCell<HashMap<MathKey, MathGraph>> = RefCell::new(HashMap::new()); }
fn bounded(dims: &[usize]) -> bool {
    dims.iter()
        .try_fold(2usize, |n, &d| n.checked_mul(d))
        .is_some_and(|n| n > 0 && n <= MAX_ACTIVATION_BYTES)
}
fn product_shape(a: &[usize], b: &[usize], transpose_rhs: bool) -> Option<Vec<usize>> {
    if !(2..=4).contains(&a.len()) || a.len() != b.len() || !bounded(a) || !bounded(b) {
        return None;
    }
    let n = a.len();
    if a[..n - 2] != b[..n - 2] || a[n - 1] != b[n - if transpose_rhs { 1 } else { 2 }] {
        return None;
    }
    let mut out = a.to_vec();
    out[n - 1] = b[n - if transpose_rhs { 2 } else { 1 }];
    bounded(&out).then_some(out)
}
pub(super) fn try_matmul(a: &Tensor, b: &Tensor, transpose_rhs: bool) -> Result<Option<Tensor>> {
    if !a.device().is_metal()
        || !a.device().same_device(b.device())
        || a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || !supported_device(a.device())
        || product_shape(a.dims(), b.dims(), transpose_rhs).is_none()
    {
        return Ok(None);
    }
    Ok(Some(a.contiguous()?.apply_op2_no_bwd(
        &b.contiguous()?,
        &MathOp::Matmul(transpose_rhs),
    )?))
}
pub(super) fn try_softmax(a: &Tensor) -> Result<Option<Tensor>> {
    if !a.device().is_metal()
        || a.dtype() != DType::BF16
        || !(2..=4).contains(&a.rank())
        || !bounded(a.dims())
        || !supported_device(a.device())
    {
        return Ok(None);
    }
    Ok(Some(a.contiguous()?.apply_op1_no_bwd(&MathOp::Softmax)?))
}
fn math_forward(
    op: MathOp,
    a: &MetalStorage,
    al: &Layout,
    b: Option<(&MetalStorage, &Layout)>,
) -> Result<(MetalStorage, Shape)> {
    let ao = offset(a, al, MAX_ACTIVATION_BYTES)?;
    let (ad, bd, out_shape, bo) = match (op, b) {
        (MathOp::Matmul(transpose), Some((b, bl))) => {
            if a.device().id() != b.device().id() {
                candle_core::bail!("MPS attention device");
            }
            let bo = offset(b, bl, MAX_ACTIVATION_BYTES)?;
            let shape = product_shape(al.dims(), bl.dims(), transpose)
                .ok_or_else(|| candle_core::Error::Msg("MPS attention dimensions".into()))?;
            (al.dims().to_vec(), bl.dims().to_vec(), shape, bo)
        }
        (MathOp::Softmax, None) => {
            let cols = *al
                .dims()
                .last()
                .ok_or_else(|| candle_core::Error::Msg("MPS softmax rank".into()))?;
            if cols == 0 || !bounded(al.dims()) {
                candle_core::bail!("MPS softmax bounds");
            }
            (
                vec![al.shape().elem_count() / cols, cols],
                vec![],
                al.dims().to_vec(),
                0,
            )
        }
        _ => candle_core::bail!("MPS attention operation"),
    };
    let n = out_shape.iter().product::<usize>();
    let device = a.device();
    device.wait_until_completed()?;
    let output = device
        .new_buffer_builder()
        .with_size_for(n, DType::BF16)
        .with_label("text-bf16-attention-mpsgraph")
        .build()?;
    autoreleasepool(|_| -> Result<()> {
        MATH_GRAPHS.with(|cache| -> Result<()> {
            let mut cache = cache.borrow_mut();
            let key = (device.id(), op, ad.clone(), bd.clone());
            if !cache.contains_key(&key) {
                if cache.len() >= MAX_GRAPHS {
                    cache.clear();
                }
                let dimensions = |ds: &[usize]| {
                    NSArray::from_retained_slice(
                        &ds.iter()
                            .map(|&d| NSNumber::new_usize(d))
                            .collect::<Vec<_>>(),
                    )
                };
                let graph = MPSGraph::new();
                let ap = unsafe { graph.placeholder(&dimensions(&ad), BF16, None) };
                let (bp, cp) = match op {
                    MathOp::Matmul(transpose) => {
                        let bp = unsafe { graph.placeholder(&dimensions(&bd), BF16, None) };
                        let right = if transpose {
                            unsafe { graph.transpose(&bp, bd.len() - 2, bd.len() - 1, None) }
                        } else {
                            bp.clone()
                        };
                        let cp = unsafe { graph.matmul(&ap, &right, None) };
                        (Some(bp), cp)
                    }
                    MathOp::Softmax => {
                        let f = unsafe { graph.cast(&ap, 0x1000_0020, None) };
                        let s = unsafe { graph.softmax(&f, -1, None) };
                        (None, unsafe { graph.cast(&s, BF16, None) })
                    }
                };
                let queue = device
                    .metal_device()
                    .as_ref()
                    .newCommandQueue()
                    .ok_or_else(|| candle_core::Error::Msg("MPS attention queue".into()))?;
                cache.insert(
                    key.clone(),
                    MathGraph {
                        graph,
                        a: ap,
                        b: bp,
                        c: cp,
                        queue,
                    },
                );
            }
            let g = &cache[&key];
            let av = unsafe { data_shape(a.buffer().as_ref(), ao, &ad) };
            let output_dims = if op == MathOp::Softmax {
                &ad
            } else {
                &out_shape
            };
            let cv = unsafe { data_shape(output.as_ref().as_ref(), 0, output_dims) };
            let results = NSDictionary::from_slices(&[&*g.c], &[&*cv]);
            if let Some((b, _)) = b {
                let bv = unsafe { data_shape(b.buffer().as_ref(), bo, &bd) };
                let bp = g
                    .b
                    .as_ref()
                    .ok_or_else(|| candle_core::Error::Msg("MPS attention placeholder".into()))?;
                let feeds = NSDictionary::from_slices(&[&*g.a, &**bp], &[&*av, &*bv]);
                unsafe {
                    g.graph.run(&g.queue, &feeds, None, &results);
                }
            } else {
                let feeds = NSDictionary::from_slices(&[&*g.a], &[&*av]);
                unsafe {
                    g.graph.run(&g.queue, &feeds, None, &results);
                }
            }
            Ok(())
        })
    })?;
    Ok((
        MetalStorage::new(output, device.clone(), n, DType::BF16),
        Shape::from(out_shape),
    ))
}
impl CustomOp2 for MathOp {
    fn name(&self) -> &'static str {
        "text-bf16-batched-mpsgraph"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("MPS attention requires Metal")
    }
    fn metal_fwd(
        &self,
        a: &MetalStorage,
        al: &Layout,
        b: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        math_forward(*self, a, al, Some((b, bl)))
    }
}
impl CustomOp1 for MathOp {
    fn name(&self) -> &'static str {
        "text-bf16-softmax-mpsgraph"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("MPS softmax requires Metal")
    }
    fn metal_fwd(&self, a: &MetalStorage, al: &Layout) -> Result<(MetalStorage, Shape)> {
        math_forward(*self, a, al, None)
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use candle_core::Device;
    use half::bf16;

    #[test]
    fn shapes_caps_and_cpu_decline() -> Result<()> {
        assert_eq!(
            product_shape(&[2, 3, 5], &[2, 5, 7], false),
            Some(vec![2, 3, 7])
        );
        assert_eq!(
            product_shape(&[2, 3, 5], &[2, 7, 5], true),
            Some(vec![2, 3, 7])
        );
        for (a, b) in [
            (vec![2, 3, 5], vec![1, 5, 7]),
            (vec![2, 0, 5], vec![2, 5, 7]),
            (vec![2, 3, 5], vec![2, 4, 7]),
            (vec![usize::MAX, 3, 5], vec![usize::MAX, 5, 7]),
        ] {
            assert!(product_shape(&a, &b, false).is_none());
        }
        assert!(!bounded(&[MAX_ACTIVATION_BYTES, 2]));
        let x = Tensor::zeros((1, 3, 5), DType::BF16, &Device::Cpu)?;
        let y = Tensor::zeros((1, 5, 7), DType::BF16, &Device::Cpu)?;
        assert!(try_matmul(&x, &y, false)?.is_none());
        assert!(try_softmax(&x)?.is_none());
        assert!(!supported_device(&Device::Cpu));
        Ok(())
    }

    #[test]
    fn batched_non_square_views_and_f64() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supported_device(&device) {
            return Ok(());
        }
        let (b, m, k, n) = (4, 3, 5, 7);
        let a = (0..b * m * k)
            .map(|i| ((i % 13) as f32 - 6.0) / 16.0)
            .collect::<Vec<_>>();
        let c = (0..b * k * n)
            .map(|i| ((i % 11) as f32 - 5.0) / 32.0)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for batch in 0..b {
            for row in 0..m {
                for col in 0..n {
                    let dot = (0..k)
                        .map(|z| {
                            f64::from(a[(batch * m + row) * k + z])
                                * f64::from(c[(batch * k + z) * n + col])
                        })
                        .sum::<f64>();
                    expected.push(bf16::from_f64(dot).to_bits());
                }
            }
        }
        let packed = |values: &[f32]| -> Result<Tensor> {
            let mut data = vec![bf16::ZERO; 7];
            data.extend(values.iter().map(|&x| bf16::from_f32(x)));
            let len = data.len();
            Tensor::from_vec(data, len, &device)?.narrow(0, 7, values.len())
        };
        let aa = packed(&a)?.reshape((2, 2, m, k))?;
        let cc = packed(&c)?.reshape((2, 2, k, n))?;
        for transpose in [false, true] {
            let rhs = if transpose {
                cc.transpose(2, 3)?
            } else {
                cc.clone()
            };
            let actual = try_matmul(&aa, &rhs, transpose)?
                .unwrap()
                .flatten_all()?
                .to_vec1::<bf16>()?;
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }
        let scores = Tensor::from_vec(
            vec![0.0f32, 0.25, -0.5, 0.5, 0.0, -0.25],
            (1, 2, 3),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let actual = try_softmax(&scores)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<bf16>()?;
        let mut expected = Vec::new();
        for row in [[0.0f64, 0.25, -0.5], [0.5, 0.0, -0.25]] {
            let exp = row.map(f64::exp);
            let sum = exp.iter().sum::<f64>();
            expected.extend(exp.map(|x| bf16::from_f64(x / sum).to_bits()));
        }
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected
        );
        assert!(try_matmul(&aa.to_dtype(DType::F32)?, &cc.to_dtype(DType::F32)?, false)?.is_none());
        assert!(try_softmax(&scores.to_dtype(DType::F32)?)?.is_none());
        Ok(())
    }
}
