# lns-ml
A high-performance LLM inference engine built on Logarithmic Number System (LNS) quantization — written from scratch in Rust.

## Quick start

```bash
cargo build --release
cargo test
```

```bash
# Inspect a .lns model
cargo run -p lns-cli -- inspect --model /path/to/model.lns

# Benchmark Q4_L decode (CPU reference)
cargo run -p lns-cli -- bench --model /path/to/model.lns --backend cpu

# Metal path (currently host API + kernel skeleton)
cargo run -p lns-cli -- bench --model /path/to/model.lns --backend metal
```
