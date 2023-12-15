import * as shaders from "./shaders/index.ts";
import { Model } from "../interfaces.ts"


export class WebGPUSimpleNeuralNet implements Model {
  private weightsInputHiddenBuffer: GPUBuffer;
  private weightsHiddenOutputBuffer: GPUBuffer;
  private biasHiddenBuffer: GPUBuffer;
  private biasOutputBuffer: GPUBuffer;
  private scalarBuffer: GPUBuffer;

  constructor(
    private device: GPUDevice,
    public params: {
      inputNodes: number;
      hiddenNodes: number;
      outputNodes: number;
      learningRate: number;
    }
  ) {
    this.weightsInputHiddenBuffer = this.createRandomBuffer(
      "weightsInputHidden",
      params.hiddenNodes * params.inputNodes
    );
    this.weightsHiddenOutputBuffer = this.createRandomBuffer(
      "weightsHiddenOutput",
      params.outputNodes * params.hiddenNodes
    );
    this.biasHiddenBuffer = this.createRandomBuffer("biasHidden", params.hiddenNodes);
    this.biasOutputBuffer = this.createRandomBuffer("biasOutput", params.outputNodes);
    this.scalarBuffer = this.createScalarBuffer([params.inputNodes, params.outputNodes]);
    this.device.pushErrorScope('validation');
  }

  // Buffer Creation

  private createBuffer(nodeCount: number): GPUBuffer {
    return this.device.createBuffer({
      size: nodeCount * 4, // Assuming each node's output is a single float
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  private createRandomBuffer(label: string, size: number) {
    const array = new Float32Array(size);
    for (let i = 0; i < array.length; ++i) {
      array[i] = this.randomNumber();
    }

    const buffer = this.device.createBuffer({
      label,
      size: array.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();

    return buffer;
  }

  private randomNumber(): number {
    return Math.random() * 2 - 1;
  }

  private createScalarBuffer(values: number[]): GPUBuffer {
    const array = new Uint32Array(values);
    const buffer = this.device.createBuffer({
        size: array.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Uint32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();
    return buffer;
  }

  // Buffer Metadata
  private bindGroupLayoutEntry(index: number) {
    return {
      binding: index,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage",
      },
    } as const;
  }

  private bindGroupEntry(index: number, buffer: GPUBuffer) {
    return {
      binding: index,
      resource: {
        buffer,
      },
    } as const;
  }

  private async readFromBuffer(buffer: GPUBuffer, size: number): Promise<number[]> {
    // Create a GPU buffer for reading back in CPU land
    const readBuffer = this.device.createBuffer({
      size: size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  
    // Encode commands for copying buffer to the readable buffer
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    this.device.queue.submit([commandEncoder.finish()]);
  
    // Wait for the GPU to finish executing before reading back the data
    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArrayBuffer = readBuffer.getMappedRange();
  
    // Copy data into a Float32Array and return it
    const data = Array.from(new Float32Array(copyArrayBuffer));
    readBuffer.unmap();
    return data;
  }


  
  public train(input: number[], output: number[]): void | Promise<void> {
    throw new Error("Method not implemented.");
  }

  public async feedForward(input: number[]): Promise<number[]> {
    const inputBuffer = this.device.createBuffer({
      label: "input buffer",
      size: input.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(inputBuffer, 0, new Float32Array(input));
    
    const initValue = new Float32Array(this.params.outputNodes).fill(0.5);
    const hiddenOutputBuffer = this.createBuffer(this.params.hiddenNodes);
    const finalOutputBuffer = this.createBuffer(this.params.outputNodes);
    this.device.queue.writeBuffer(hiddenOutputBuffer, 0, initValue);
    this.device.queue.writeBuffer(finalOutputBuffer, 0, initValue);
    
  
    // First Dispatch: Input to Hidden Layer
    this.dispatch(
      inputBuffer, 
      this.weightsInputHiddenBuffer,
      this.biasHiddenBuffer,
      hiddenOutputBuffer,
      Math.ceil(this.params.hiddenNodes / 64)
      );
  
    // Second Dispatch: Hidden Layer to Output Layer
    this.dispatch(
      hiddenOutputBuffer,
      this.weightsHiddenOutputBuffer,
      this.biasOutputBuffer,
      finalOutputBuffer,
      Math.ceil(this.params.outputNodes / 64)
      );
  
      await this.device.queue.onSubmittedWorkDone();
      const outputArrayBuffer = await this.readFromBuffer(finalOutputBuffer, this.params.outputNodes * 4);
      return Array.from(outputArrayBuffer);
  }
  

  private dispatch(inputBuffer: GPUBuffer, weightBuffer: GPUBuffer, biasBuffer: GPUBuffer, outputBuffer: GPUBuffer, workgroupCount: number) {
    const { device } = this;

    const bufferBindGroupLayout = device.createBindGroupLayout({
      entries: [
        this.bindGroupLayoutEntry(0),
        this.bindGroupLayoutEntry(1),
        this.bindGroupLayoutEntry(2),
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
      ],
    });
    const bufferbindGroup = device.createBindGroup({
      label: "feedForward bind group",
      layout: bufferBindGroupLayout,
      entries: [
        this.bindGroupEntry(0, inputBuffer),
        this.bindGroupEntry(
          1,
          weightBuffer,
        ),
        this.bindGroupEntry(
          2,
          biasBuffer,
        ),
        this.bindGroupEntry(3, outputBuffer),
      ],
    });

    const scalarBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" }
        }
      ],
    });
    const scalarBindGroup = device.createBindGroup({
      layout: scalarBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.scalarBuffer,
          },
        }
      ],
    });

    const pipeline = device.createComputePipeline({
      label: "feedForward pipeline",
      layout: device.createPipelineLayout({
        label: "feedForward pipeline layout",
        bindGroupLayouts: [bufferBindGroupLayout, scalarBindGroupLayout],
      }),
      compute: {
        module: device.createShaderModule({
          label: "feedForward shader",
          code: shaders.feedForward,
        }),
        entryPoint: "main",
      },
    });
    
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bufferbindGroup);
    pass.setBindGroup(1, scalarBindGroup);
    
    pass.insertDebugMarker("dispatch");
    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  static async create(params: {
    inputNodes: number;
    hiddenNodes: number;
    outputNodes: number;
    learningRate: number;
  }) {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    const device = await adapter?.requestDevice();
    if (!device) throw new Error("no suitable adapter found");
    device.lost.then((e) => console.error(e)); 
    return new WebGPUSimpleNeuralNet(device, params);
  }
}
