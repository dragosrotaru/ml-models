// WGSL Compute Shader for Feedforward Neural Network

// Define the size of the workgroup
const workgroupSize = 64;

// Buffers for inputs, weights, biases, and outputs
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, write> outputs: array<f32>;

// Number of input and output nodes (to be set from TypeScript)
@group(1) @binding(0) var inputNodes: u32;
@group(1) @binding(1) var outputNodes: u32;

// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Compute shader main function
@compute @workgroup_size(workgroupSize)
fn maind(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Calculate the index of the output node being processed
    let idx = global_id.x;

    // Compute the weighted sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < inputNodes; i++) {
        sum = sum + inputs[i] * weights[i * outputNodes + idx];
    }

    // Add the bias and apply the sigmoid function
    outputs[idx] = sigmoid(sum + biases[idx]);
}

@compute @workgroup_size(workgroupSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < outputNodes) {
        outputs[idx] = inputs[idx % inputNodes]; // Directly pass input to output for testing
    }
}