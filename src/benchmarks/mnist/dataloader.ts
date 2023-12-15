export class MNISTDataloader {
  constructor(
    private trainingImagesFilepath: string,
    private trainingLabelsFilepath: string,
    private testImagesFilepath: string,
    private testLabelsFilepath: string
  ) {}

  private async loadLabels(filepath: string): Promise<number[]> {
    const fileBuffer = await Deno.readFile(filepath);
    const view = new DataView(fileBuffer.buffer);

    const magic = view.getInt32(0, false);
    if (magic !== 2049) throw new Error(`Magic number mismatch, expected 2049, got ${magic}`);

    return Array.from(new Uint8Array(fileBuffer.buffer, 8));
  }

  private async loadImages(filepath: string): Promise<number[][][]> {
    const fileBuffer = await Deno.readFile(filepath);
    const view = new DataView(fileBuffer.buffer);

    const magic = view.getInt32(0, false);
    const size = view.getInt32(4, false);
    const rows = view.getInt32(8, false);
    const cols = view.getInt32(12, false);

    if (magic !== 2051) throw new Error(`Magic number mismatch, expected 2051, got ${magic}`);

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

  private async loadDataset(imagesFilepath: string, labelsFilepath: string) {
    const labels = await this.loadLabels(labelsFilepath);
    const images = await this.loadImages(imagesFilepath);
    return { images, labels };
  }

  public async loadAll() {
    return {
      train: await this.loadDataset(this.trainingImagesFilepath, this.trainingLabelsFilepath),
      test: await this.loadDataset(this.testImagesFilepath, this.testLabelsFilepath),
    };
  }
}
