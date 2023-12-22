import { Model } from "../../interfaces.ts";
import { MNISTDataloader } from "./dataloader.ts";
import { TFNeuralNet } from "../../models/tf-neural-net/index.ts";
import * as tf from "npm:@tensorflow/tfjs";

export class MNISTBenchmark {
  private DATA_FOLDER = "src/benchmarks/mnist/data/";
  public data = new MNISTDataloader(
    this.DATA_FOLDER + "train/images",
    this.DATA_FOLDER + "train/labels",
    this.DATA_FOLDER + "test/images",
    this.DATA_FOLDER + "test/labels"
  ).loadAll();

  constructor(private model: Model | TFNeuralNet) {}

  private normalizeImage(image: number[][]) {
    return image.flat().map((p) => p / 255);
  }

  private normalizeLabel(label: number): number[] {
    const expectedOutput = Array(this.model.params.outputNodes).fill(0);
    expectedOutput[label] = 1;
    return expectedOutput;
  }

  public async train(epochCount: number) {
    const { images, labels } = (await this.data).train;

    // TFJS
    const batchSize = 100;
    if (this.model instanceof TFNeuralNet) {
      for (let i = 0; i <= 10000; i += batchSize) {
        const batchImages = images.slice(i, i + batchSize);
        const batchLabels = labels.slice(i, i + batchSize);
        console.log(`${i + 1} of ${images.length}`);

        // normalize data
        const input = tf.tidy(() => tf.tensor2d(batchImages.map(this.normalizeImage)));
        const output = tf.tidy(() => tf.tensor2d(batchLabels.map((label) => this.normalizeLabel(label))));

        await this.model.model.fit(input, output, {
          epochs: epochCount,
          batchSize,
        });

        // Dispose tensors to free memory
        input.dispose();
        output.dispose();
      }
      return;
    }

    // Model
    for (let epoch = 1; epoch <= epochCount; epoch++) {
      for (let i = 0; i < images.length; i++) {
        console.log(`${i + 1} of ${images.length}`);
        const image = images[i];
        const label = labels[i];

        // normalize data
        const input = this.normalizeImage(image);
        const output = this.normalizeLabel(label);

        await this.model.train(input, output);
      }
      console.log(`Epoch ${epoch} complete.`);
    }
  }

  public async test(): Promise<number> {
    const { images, labels } = (await this.data).test;

    // TFJS
    if (this.model instanceof TFNeuralNet) {
      // normalize data
      const input = images.map(this.normalizeImage);
      const output = labels.map((label) => this.normalizeLabel(label));

      const result = await this.model.model.evaluate(tf.tensor2d(input), tf.tensor2d(output));
      return Array.isArray(result) ? (await result[1].data())[0] : (await result.dataSync())[0];
    }

    // Models
    let correctCount = 0;

    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      const label = labels[i];

      // normalize data
      const input = this.normalizeImage(image);
      const output = await this.model.feedForward(input);
      const predicted = output.indexOf(Math.max(...output));
      // console.log(predicted, label);

      if (predicted === label) correctCount++;
    }

    return correctCount / images.length;
  }
}
