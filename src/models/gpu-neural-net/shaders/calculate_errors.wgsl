// WGSL Compute Shader for Error Propagation

const workgroupSize = 128;

struct Uniforms {
    inputNodeCount: f32,
    nodeCount: f32,
    learningRate: f32, // unused
    nextLayerNodeCount: f32, // 0 = output layer, n = hidden layer with n nodes in the next layer
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(0) @binding(1) var<storage, read_write> inputs: array<f32>; // unused
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>; // input
@group(0) @binding(3) var<storage, read_write> errors: array<f32>; // output

@group(0) @binding(4) var<storage, read_write> weights: array<f32>; // unused
@group(0) @binding(5) var<storage, read_write> biases: array<f32>; // unused

@group(0) @binding(6) var<storage, read_write> nextLayerWeights: array<f32>;       // Weights to the next layer, for hidden layers
@group(0) @binding(7) var<storage, read_write> targetsOrNextLayerErrors: array<f32>; // Targets (output layer) or Errors (hidden layers)


@compute @workgroup_size(workgroupSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let nodeCount = u32(uniforms.nodeCount); // number of nodes in this layer 
    let nextLayerNodeCount = u32(uniforms.nextLayerNodeCount); // number of nodes in the next layer
    let idx = global_id.x; // the index of the node in this layer

    var error: f32 = 0.0;
    if (nextLayerNodeCount == 0u) {
        // Output layer error calculation
        error = targetsOrNextLayerErrors[idx] - outputs[idx];
    } else {
        // Hidden layer error calculation
        for (var j: u32 = 0u; j < nextLayerNodeCount; j = j + 1u) {
            error += targetsOrNextLayerErrors[j] * nextLayerWeights[j * nodeCount + idx];
        }
    }
    errors[idx] = error;
}
