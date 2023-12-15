// WGSL Compute Shader for Feedforward Neural Network
const workgroupSize = 32;

// Shared memory for inputs
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> transposedWeights: array<f32>;  // Transposed weights
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, write> outputs: array<f32>;

@group(1) @binding(0) var inputNodes: u32;
@group(1) @binding(1) var outputNodes: u32;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(workgroupSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Compute the weighted sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < inputNodes; i++) {
        sum += inputs[i] * transposedWeights[idx * inputNodes + i];  // Accessing transposed weights
    }

    // Add the bias and apply the sigmoid function
    outputs[idx] = sigmoid(sum + biases[idx]);
}