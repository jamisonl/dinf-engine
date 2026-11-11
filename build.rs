fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/diffusion/rope_sincos.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }
    cc::Build::new()
        .file("src/diffusion/rope_sincos.c")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .compile("dinf_rope_sincos");
}
