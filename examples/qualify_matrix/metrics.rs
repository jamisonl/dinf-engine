use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct Row {
    pub row: usize,
    pub reference_top: usize,
    pub actual_top: usize,
    pub teacher_token: u32,
    pub kl: f64,
    pub max_abs: f64,
    pub rms: f64,
    pub reference_nll: f64,
    pub actual_nll: f64,
    pub nll_delta: f64,
}

fn log_normalizer(values: &[f32]) -> (f64, f64) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum = values
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>();
    (max, sum.ln())
}

pub(super) fn compare(row: usize, reference: &[f32], actual: &[f32], teacher: u32) -> Result<Row> {
    ensure!(
        !reference.is_empty() && reference.len() == actual.len(),
        "logit shape differs"
    );
    ensure!(
        (teacher as usize) < reference.len(),
        "teacher outside vocabulary"
    );
    ensure!(
        reference.iter().chain(actual).all(|v| v.is_finite()),
        "non-finite logits"
    );
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
    };
    let (mr, zr) = log_normalizer(reference);
    let (ma, za) = log_normalizer(actual);
    let mut kl = 0.0;
    let mut max_abs: f64 = 0.0;
    let mut squares = 0.0;
    for (&r, &a) in reference.iter().zip(actual) {
        let (r, a) = (f64::from(r), f64::from(a));
        let log_r = (r - mr) - zr;
        let log_a = (a - ma) - za;
        kl += log_r.exp() * (log_r - log_a);
        max_abs = max_abs.max((r - a).abs());
        squares += (r - a).powi(2);
    }
    let reference_nll = zr - (f64::from(reference[teacher as usize]) - mr);
    let actual_nll = za - (f64::from(actual[teacher as usize]) - ma);
    Ok(Row {
        row,
        reference_top: argmax(reference),
        actual_top: argmax(actual),
        teacher_token: teacher,
        kl: kl.max(0.0),
        max_abs,
        rms: (squares / actual.len() as f64).sqrt(),
        reference_nll,
        actual_nll,
        nll_delta: actual_nll - reference_nll,
    })
}

pub(super) fn passed(matches: usize, rows: usize, mean_kl: f64, max_kl: f64) -> bool {
    rows == 128
        && matches <= rows
        && matches as f64 / rows as f64 >= 0.95
        && mean_kl.is_finite()
        && max_kl.is_finite()
        && (0.0..=0.02).contains(&mean_kl)
        && (0.0..=0.1).contains(&max_kl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_kl_and_teacher_nll_have_the_declared_direction() -> Result<()> {
        let row = compare(0, &[3.0f32.ln(), 0.0], &[0.0, 0.0], 1)?;
        let expected = 0.75f64 * 1.5f64.ln() + 0.25 * 0.5f64.ln();
        assert!((row.kl - expected).abs() < 1e-7);
        assert!((row.reference_nll - 4.0f64.ln()).abs() < 1e-7);
        assert!((row.nll_delta + 2.0f64.ln()).abs() < 1e-7);
        assert_eq!(row.reference_top, 0);
        assert_eq!(row.teacher_token, 1);
        assert!(compare(0, &[0.0], &[f32::NAN], 0).is_err());
        assert!(compare(0, &[0.0], &[0.0], 1).is_err());
        Ok(())
    }

    #[test]
    fn gates_require_122_of_128_and_reject_partial_or_nonfinite_evidence() {
        assert!(passed(122, 128, 0.02, 0.1));
        assert!(!passed(121, 128, 0.0, 0.0));
        assert!(!passed(127, 127, 0.0, 0.0));
        assert!(!passed(128, 128, 0.02000001, 0.0));
        assert!(!passed(128, 128, 0.0, 0.10000001));
        assert!(!passed(128, 128, f64::NAN, 0.0));
    }
}
