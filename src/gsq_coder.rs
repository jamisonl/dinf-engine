//! Immutable public GSQ RCO Coder artifact identity; text-only native profile.
//! Source: ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF.
pub const REVISION: &str = "54229d38d588a66d1c6cff08af7b281b9326c656";
pub const MODEL_ID: &str = "deepseek-v4.1-flash-1017-gsq-rco-coder";
pub const PREFIX: &str = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS";

pub(crate) fn is_entry(path: &std::path::Path) -> bool {
    path.file_name().and_then(|s| s.to_str())
        == Some(format!("{PREFIX}-00001-of-00006.gguf").as_str())
}
/// In canonical split order. The upstream Engram asset is installed as part six.
pub const PARTS: [(u64, &str); 6] = [
    (
        5_780_704,
        "108e63d27240da2d99846250727a7b282e05a7fdf2448435e81904b9a98d8a4a",
    ),
    (
        19_776_064_960,
        "75079999eb3a492ad695f3466fd6a068a1cf1920980eac1c7f51406363fadf7a",
    ),
    (
        19_939_989_696,
        "ea964e525922f36d3b984299c1a5bb8d896f50b9c2942364cd221c37863331f2",
    ),
    (
        19_682_533_472,
        "2d95806d186597cb0ea32635c92f0b66a9a6afae1a399ff74b13aebaa4ac7a52",
    ),
    (
        12_719_198_624,
        "3665b48c3bd46330780caf286ee40a087ac11499e5b7df8c943b098d10f426d4",
    ),
    (
        110_595_290_816,
        "3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3",
    ),
];
