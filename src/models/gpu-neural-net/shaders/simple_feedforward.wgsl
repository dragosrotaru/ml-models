// WGSL Compute Shader for Feedforward Neural Network

// Define the size of the workgroup
const workgroupSize = 128;

struct Uniforms {
    inputLength: f32,
    hiddenLength: f32,
    outputLength: f32,
    learningRate: f32,
};

// Buffers for inputs, weights, biases, and outputs
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> biases: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;

// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Compute shader main function
@compute @workgroup_size(workgroupSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Calculate the index of the output node being processed
    let idx = global_id.x;
    let inputLength = u32(uniforms.inputLength);
    let outputLength = u32(uniforms.outputLength);

    // Compute the weighted sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < inputLength; i = i + 1u) {
        sum = sum + inputs[i] * weights[i * outputLength + idx];
    }

    // Add the bias and apply the sigmoid function
    outputs[idx] = sigmoid(sum + biases[idx]);
}