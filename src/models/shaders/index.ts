export const feedForward = Deno.readTextFileSync("./src/models/shaders/simple_feedForward.wgsl");
export const backpropHidden = Deno.readTextFileSync("./src/models/shaders/backprop_hidden.wgsl");
export const backpropOutput = Deno.readTextFileSync("./src/models/shaders/backprop_output.wgsl");
