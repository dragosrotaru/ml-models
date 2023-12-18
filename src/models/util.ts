/**
 *
 * @returns random number in the range [-1, 1]
 */
export const randomNumber = (): number => Math.random() * 2 - 1;
/**
 *
 * @returns array of random numbers in the range [-1, 1] of given length
 */
export const randomArray = (length: number): number[] => Array.from({ length }, randomNumber);

/**
 * Sigmoid Function
 *
 * @description
 * smooth bounded, differentiable, real function
 * that is defined for all real input values and
 * has a non-negative derivative at each point and exactly one inflection point
 * Latex: sigmoid(x) = 1 / (1 + e^{-x}).
 * Has a steep change around x = 0 and asymptotic behavior towards 0 (for x → -∞) and 1 (for x → ∞)
 *
 * https://en.wikipedia.org/wiki/Sigmoid_function
 *
 * @param {number} x - The input value.
 * @returns {number} The output between 0 and 1.
 */
export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

/**
 * Returns the derivative of the sigmoid function
 */
export const sigmoidDerivative = (y: number): number => y * (1 - y);
