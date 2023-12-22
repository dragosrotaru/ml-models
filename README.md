# ML Benchmarks

This repository contains various Machine Learning models implemented using different languages, backends and optimization techniques. The purpose is to benchmark the performance characteristics of alternative technologies.

### Technologies

Currently Supported:
- WebGPU
- Rust
- Deno (TypeScript)
- Tensorflow.js

To be Added:
- TensorFlow
- WASM
- PyTorch
- Jax
- Haskell
- Lisp

### Models and Benchmarks

- MNIST
- Multi Layer Perceptron

### Useage

Run the benchmarks using:

`deno run --unstable --allow-all src/index.ts`
`cargo run --release`

### Todo

- implement using other frameworks/languages
- compile rust to WASM
- implement a more robust benchmarking / debugging harness
- record results
