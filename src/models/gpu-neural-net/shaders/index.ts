export const feedForward = Deno.readTextFileSync("./src/models/gpu-neural-net/shaders/simple_feedForward.wgsl");
export const backprop = Deno.readTextFileSync("./src/models/gpu-neural-net/shaders/backprop.wgsl");
