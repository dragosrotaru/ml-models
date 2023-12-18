export const feedForward = Deno.readTextFileSync("./src/models/gpu-neural-net/shaders/feedforward.wgsl");
export const updateWeightsAndBiases = Deno.readTextFileSync(
  "./src/models/gpu-neural-net/shaders/update_weights_biases.wgsl"
);
export const calculateErrors = Deno.readTextFileSync("./src/models/gpu-neural-net/shaders/calculate_errors.wgsl");
