import { SimpleNeuralNet } from "../../models/simpleNeuralNet";
import { MNISTDataloader } from "./dataloader";

export class MNISTBenchmark {
  private DATA_FOLDER = "src/benchmarks/mnist/data/";
  public data = new MNISTDataloader(
    this.DATA_FOLDER + "train/images",
    this.DATA_FOLDER + "train/labels",
    this.DATA_FOLDER + "test/images",
    this.DATA_FOLDER + "test/labels"
  ).loadAll();

  constructor(private neuralNet: SimpleNeuralNet) {
  }

  private normalizeImage(image: number[][]) {
    return image.flat().map((p) => p / 255)
  }

  private normalizeLabel(label: number): number[] {
    const expectedOutput = Array(this.neuralNet.outputNodes).fill(0);
    expectedOutput[label] = 1;
    return expectedOutput;
  }

  public train(epochCount: number): void {
    const { images, labels } = this.data.train;
    for (let epoch = 1; epoch <= epochCount; epoch++) {
        for (let i = 0; i < images.length; i++) {
            console.log(`${i + 1} of ${images.length}`);
            const image = images[i];
            const label = labels[i];

            // normalize data
            const input = this.normalizeImage(image);
            const output = this.normalizeLabel(label);

            this.neuralNet.train(input, output);
            
        }
      console.log(`Epoch ${epoch} complete.`);
    }
  }

  public test(): number {
    const { images, labels } = this.data.test;
    let correctCount = 0;

    for (let i = 0; i < images.length; i++) {
        const image = images[i];
        const label = labels[i];

        // normalize data
        const input = this.normalizeImage(image);
        const output = this.neuralNet.feedForward(input);
        const predicted = output.indexOf(Math.max(...output));

        if (predicted === label) correctCount++;
        
    }

    return correctCount / images.length;

    }
}
