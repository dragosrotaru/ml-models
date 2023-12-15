export const feedForward = Deno.readTextFileSync("./src/models/shaders/simple_feedForward.wgsl");
export const backprop = Deno.readTextFileSync("./src/models/shaders/backprop.wgsl");
