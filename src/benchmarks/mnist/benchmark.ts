import { Model } from "../../interfaces.ts";
import { MNISTDataloader } from "./dataloader.ts";

export class MNISTBenchmark {
  private DATA_FOLDER = "src/benchmarks/mnist/data/";
  public data = new MNISTDataloader(
    this.DATA_FOLDER + "train/images",
    this.DATA_FOLDER + "train/labels",
    this.DATA_FOLDER + "test/images",
    this.DATA_FOLDER + "test/labels"
  ).loadAll();

  constructor(private neuralNet: Model) {}

  private normalizeImage(image: number[][]) {
    return image.flat().map((p) => p / 255);
  }

  private normalizeLabel(label: number): number[] {
    const expectedOutput = Array(this.neuralNet.params.outputNodes).fill(0);
    expectedOutput[label] = 1;
    return expectedOutput;
  }

  public async train(epochCount: number) {
    const { images, labels } = (await this.data).train;
    const sample = 60000; // images.length;
    for (let epoch = 1; epoch <= epochCount; epoch++) {
      for (let i = 0; i < sample; i++) {
        console.log(`${i + 1} of ${images.length}`);
        const image = images[i];
        const label = labels[i];

        // normalize data
        const input = this.normalizeImage(image);
        const output = this.normalizeLabel(label);

        await this.neuralNet.train(input, output);
      }
      console.log(`Epoch ${epoch} complete.`);
    }
  }

  public async test(): Promise<number> {
    const { images, labels } = (await this.data).test;
    const sample = 100; // images.length;
    let correctCount = 0;

    for (let i = 0; i < sample; i++) {
      const image = images[i];
      const label = labels[i];

      // normalize data
      const input = this.normalizeImage(image);
      const output = await this.neuralNet.feedForward(input);
      const predicted = output.indexOf(Math.max(...output));
      // console.log(predicted, label);

      if (predicted === label) correctCount++;
    }

    return correctCount / sample;
  }
}
