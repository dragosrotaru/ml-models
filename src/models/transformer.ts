const seed = 42; // todo support for reproducibility

/* Dataset */

const dataset = "WHat is the meaning of life? The meaning of life is 42.";

/* Encoding */

// Character-wise encoding
const uniqueChars = [...new Set(dataset.split(""))].sort();
const vocabSize = uniqueChars.length;

const stoi = new Map(uniqueChars.map((char, i) => [char, i]));
const itos = new Map(uniqueChars.map((char, i) => [i, char]));
const encode = (text: string) => text.split("").map((char) => stoi.get(char) as number);
const decode = (encoded: number[]) => encoded.map((i) => itos.get(i)).join("");

/* Encode Dataset */

const encoded = encode(dataset);
const n = Math.floor(encoded.length * 0.9);
const trainEncoded = encoded.slice(0, n);
const validationEncoded = encoded.slice(n);

const blockSize = 8; // context size

/**
 * Creates a dataset by moving a window of size block_size over the data
 * and creating a training example for each position in the data.
 * Example:
 * data = [1, 2, 3, 4, 5, 6, 7, 8]
 * block_size = 3
 * [1, 2, 3] -> [4], [1] -> 2, [3, 4] -> 5
 * @param data
 * @returns
 */
const createDataset = (data: number[]) => {
  const x = [];
  const y = [];
  for (let i = 0; i < data.length - blockSize; i++) {
    for (let j = 1; j <= blockSize; j++) {
      x.push(data.slice(i, i + j));
      y.push(data[i + j + 1]);
    }
  }
  return { x, y };
};

const trainDataset = createDataset(trainEncoded);
const validationDataset = createDataset(validationEncoded);

/* Batching */

const batchSize = 4; // how many independent sequences will we process in parallel

function randomInt(max: number): number {
  return Math.floor(Math.random() * max);
}

// todo can be done to include all possible sequences
function getBatch(dataset: { x: number[][]; y: number[] }, batchSize: number): { x: number[][]; y: number[] } {
  const batchX = [];
  const batchY = [];

  for (let i = 0; i < batchSize; i++) {
    const index = randomInt(dataset.x.length);
    batchX.push(dataset.x[index]);
    batchY.push(dataset.y[index]);
  }

  return { x: batchX, y: batchY };
}

/* Model */

/* 

For self-attention you need to have the tokens glean information from each other, but specifically to the 
tokens in the past. Because this is a predictive model, we cant have info from the future.
In order to do that, an easy way is to take the average of the past tokens. You can do that
using matrix multiplication with a triangular averaging matrix like this:
[1 0 0]
[1/2 1/2 0]
[1/3 1/3 1/3]

this can be generated using softmax on a triangular matrix of ones with a -inf diagonal


Head Self Attention is a system for making nodes in a graph pay attention to each other.
the nodes in the case of a transformer are the tokens in the sequence.

In the context of processing code, the important thing is to define the graph in a way which
most accurately represents the problem. In the case of code, the graph is the AST.

Instead of predicting the next token, we can predict the execution outcome of the code.
If we predict the execution model, we dont need to mask the future tokens, because we 
are not predicting the code, but the execution model.

cross attention is used to make the nodes in one graph pay attention to the nodes in another graph.


*/
