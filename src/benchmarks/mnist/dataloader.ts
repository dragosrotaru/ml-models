import * as fs from "fs";

export class MNISTDataloader {
  constructor(
    private trainingImagesFilepath: string,
    private trainingLabelsFilepath: string,
    private testImagesFilepath: string,
    private testLabelsFilepath: string
  ) {}

  private loadLabels(filepath: string): number[] {
    const fileBuffer = fs.readFileSync(filepath);
    const magic = fileBuffer.readInt32BE(0);
    if (magic !== 2049)
      throw new Error(`Magic number mismatch, expected 2049, got ${magic}`);

    return Array.from(fileBuffer.slice(8));
  }

  private loadImages(filepath: string): number[][][] {
    const fileBuffer = fs.readFileSync(filepath);
    const magic = fileBuffer.readInt32BE(0);
    const size = fileBuffer.readInt32BE(4);
    const rows = fileBuffer.readInt32BE(8);
    const cols = fileBuffer.readInt32BE(12);

    if (magic !== 2051)
      throw new Error(`Magic number mismatch, expected 2051, got ${magic}`);

    const images: number[][][] = [];

    let index = 16;
    for (let i = 0; i < size; i++) {
      images[i] = [];
      for (let row = 0; row < rows; row++) {
        images[i][row] = [];
        for (let col = 0; col < cols; col++) {
          images[i][row][col] = fileBuffer[index++];
        }
      }
    }
    return images;
  }

  private loadDataset(
    imagesFilepath: string,
    labelsFilepath: string
  ) {
    const labels = this.loadLabels(labelsFilepath);
    const images = this.loadImages(imagesFilepath);
    return { images, labels };
  }

  public loadAll() {
    return {
      train: this.loadDataset(
        this.trainingImagesFilepath,
        this.trainingLabelsFilepath
      ),
      test: this.loadDataset(this.testImagesFilepath, this.testLabelsFilepath),
    };
  }
}

