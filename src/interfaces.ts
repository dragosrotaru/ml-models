export interface Model {
    params: {
        inputNodes: number;
        hiddenNodes: number;
        outputNodes: number;
        learningRate: number;
    };
    feedForward(input: number[]): Promise<number[]> | number[];
    train(input: number[], output: number[]): Promise<void> | void;
}