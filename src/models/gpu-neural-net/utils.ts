import { randomNumber } from "../util.ts";
const DEBUG = "validation"; // out-of-memory, validation

export class GPUUtil {
  constructor(private device: GPUDevice) {}

  public readFromBuffer = async (buffer: GPUBuffer) => {
    const { device } = this;
    device.pushErrorScope(DEBUG);
    await device.queue.onSubmittedWorkDone();
    const readBuffer = device.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Encode commands for copying buffer to the readable buffer
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
    device.queue.submit([commandEncoder.finish()]);

    // Wait for the GPU to finish executing before reading back the data
    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArrayBuffer = readBuffer.getMappedRange();

    // Copy data into a Float32Array and return it
    const data = Array.from(new Float32Array(copyArrayBuffer));
    readBuffer.unmap();
    await device.queue.onSubmittedWorkDone();
    await this.popErrorScope();
    return data;
  };

  public logBuffer = async (buffer: GPUBuffer) => {
    const data = await this.readFromBuffer(buffer);
    console.log(data);
  };

  public createBuffer(size: number): GPUBuffer {
    return this.device.createBuffer({
      size, // Assuming each node's output is a single float
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  public pushErrorScope = async () => {
    // push error scope
    await this.device.pushErrorScope(DEBUG);
  };

  public popErrorScope = async () => {
    // pop error scope
    const err = await this.device.popErrorScope();
    if (err) console.error(err);
  };

  public createUniformBuffer(values: number[]): GPUBuffer {
    const array = new Float32Array(values);
    const buffer = this.device.createBuffer({
      label: "uniform buffer",
      size: array.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();
    return buffer;
  }

  public createRandomBuffer(label: string, size: number) {
    const array = new Float32Array(size);
    for (let i = 0; i < array.length; ++i) {
      array[i] = randomNumber();
    }

    const buffer = this.device.createBuffer({
      label,
      size: array.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();

    return buffer;
  }

  public createAndWriteBuffer(data: number[], label: string) {
    const buffer = this.device.createBuffer({
      label,
      size: data.length * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  }
}
