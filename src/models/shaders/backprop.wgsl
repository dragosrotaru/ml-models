// Backpropagation Shader for Hidden Layer

// Define the size of the workgroup
const workgroupSize = 64;

struct Uniforms {
    inputLength: u32,
    hiddenLength: u32,
    outputLength: u32,
    learningRate: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> outputs: array<f32>;
@group(0) @binding(3) var<storage, read> output_errors: array<f32>;
@group(0) @binding(4) var<storage, read_write> hidden_errors: array<f32>;
@group(0) @binding(5) var<storage, read_write> weights: array<f32>;
@group(0) @binding(6) var<storage, read_write> biases: array<f32>;

fn sigmoid_derivative(x: f32) -> f32 {
    return x * (1.0 - x);
}

@compute @workgroup_size(workgroupSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let inputLength = uniforms.inputLength;
    let hiddenLength = uniforms.hiddenLength;
    let outputLength = uniforms.outputLength;
    let learningRate = uniforms.learningRate;

    let output = outputs[idx];
    
    var error: f32 = 0.0;
    for (var j: u32 = 0u; j < outputLength; j = j + 1u) {
        error += output_errors[j] * weights[j * hiddenLength + idx];
    }
    hidden_errors[idx] = error;

    // calculate gradient
    let gradient = error * sigmoid_derivative(output) * learningRate;

    // Update weights and biases for output layer
    for (var i: u32 = 0u; i < inputLength;  i = i + 1u) {
        let weightIndex = idx * inputLength + i;
        weights[weightIndex] += inputs[i] * gradient;
    }
    biases[idx] += gradient;
}