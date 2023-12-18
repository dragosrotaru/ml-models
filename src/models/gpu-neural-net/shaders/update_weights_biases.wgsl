// WGSL Compute Shader for Backpropagation

const workgroupSize = 128;

struct Uniforms {
    inputNodeCount: f32,
    nodeCount: f32,
    learningRate: f32, // unused
    nextLayerNodeCount: f32, // 0 = output layer, n = hidden layer with n nodes in the next layer
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Inputs
@group(0) @binding(1) var<storage, read_write> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> errors: array<f32>;

// Outputs
@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> biases: array<f32>;

// Unused
@group(0) @binding(6) var<storage, read_write> nextLayerWeights: array<f32>;
@group(0) @binding(7) var<storage, read_write> targetsOrNextLayerErrors: array<f32>;


fn sigmoid_derivative(x: f32) -> f32 {
    return x * (1.0 - x);
}

@compute @workgroup_size(workgroupSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let inputNodeCount = u32(uniforms.inputNodeCount); // number of nodes in the input layer    
    let nodeCount = u32(uniforms.nodeCount); // number of nodes in this layer 
    let learningRate = uniforms.learningRate; // the learning rate

    let idx = global_id.x; // the index of the node in this layer
    let output = outputs[idx]; // output of this node
    let error = errors[idx]; // error of this node

    // calculate gradient
    let gradient = error * sigmoid_derivative(output) * learningRate;
    
    // update weights
    for (var i: u32 = 0u; i < inputNodeCount; i = i + 1u) {
        let weightIndex = idx * inputNodeCount + i;
        weights[weightIndex] += inputs[i] * gradient;
    }

    // update biases
    biases[idx] += gradient;
}
