# Reference prompts

`long-natural.txt` is a user message of about 1,300 tokens for
`examples/qualify_reference.rs`. It is longer than Gemma 4's 1,024-token
sliding window, and its continuation is ordinary prose, so most scored rows
carry information.

Build a comparison from it:

```sh
cargo run --release --example qualify_reference -- \
  --emit-prompt MODEL.gguf fixtures/reference-prompts/long-natural.txt > prompt.tokens
llama_reference MODEL.gguf BACKEND prompt.tokens 32 ref MODEL_SHA256
llama_reference MODEL.gguf cpu prompt.tokens 32 ref-cpu MODEL_SHA256
```

Then add the second run to `ref.json`:

```json
"second_reference_logits_file": "ref-cpu.f32",
"second_reference_backend": "cpu"
```

With a second reference, a row may differ from the first by as much as the
second does, and the mean by as much as the second's mean, where that
exceeds the fixed limits (mean KL 0.02, row KL 0.1). The fixed limits are
the floor. Use it for models whose two reference backends already disagree
beyond the fixed limits; Gemma 4 at Q4_K_M is one.

Do not build a long prompt by repeating a short one. The model answers a
looped prompt with a loop of its own: nearly every scored row is certain and
compares as zero, and the few that are not are flat distributions that move
with any change in rounding.
