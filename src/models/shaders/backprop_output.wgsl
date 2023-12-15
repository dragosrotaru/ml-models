// Backpropagation Shader for Output Layer

// Define the size of the workgroup
const workgroupSize = 64;

@group(0) @binding(0) var<storage, read> outputs: array<f32>;
@group(0) @binding(1) var<storage, read> expected: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> biases: array<f32>;
@group(0) @binding(4) var<storage, read> inputs: array<f32>;
@group(0) @binding(5) var<storage, write> errors: array<f32>;

@group(1) @binding(0) var outputNodes: u32;
@group(1) @binding(1) var inputNodes: u32;
@group(1) @binding(2) var learningRate: f32;

fn sigmoid_derivative(x: f32) -> f32 {
    return x * (1.0 - x);
}

@compute @workgroup_size(workgroupSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < outputNodes) {
        let output = outputs[idx];

        // calculate error
        let error = expected[idx] - output;
        errors[idx] = error;

        // calculate gradient
        let gradient = error * sigmoid_derivative(output) * learningRate;

        // Update weights and biases for output layer
        for (var i: u32 = 0; i < inputNodes; i++) {
            let weightIndex = idx * inputNodes + i;
            weights[weightIndex] += inputs[i] * gradient;
        }
        biases[idx] += gradient;
    }
}