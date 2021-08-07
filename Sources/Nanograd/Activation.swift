import Foundation
import Accelerate

func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
}

func sigmoid_deriv(_ x: Float) -> Float {
    return sigmoid(x) * (1.0 - sigmoid(x))
}

class Activation: Layer {
    var inputCache: Matrix
    var inputSize: Int
    var outputSize: Int
    var dx: Matrix

    init(inputSize inDim: Int, outputSize outDim: Int) {
        inputSize = inDim
        outputSize = outDim
        inputCache = Matrix(rows: 1, cols: inputSize)
        dx = Matrix(rows: 1, cols: inputSize)
    }

    override func forward(withInput input: Matrix) -> Matrix {
        withUnsafeMutablePointer(to: &inputCache.data[0]) { incachepointer in
            withUnsafePointer(to: &input.data[0]) { inpointer in
                // inputCache := input
                cblas_scopy(Int32(input.cols), inpointer, 1, incachepointer, 1)
            }
        }

        return input.map(withFunction: sigmoid)
    }

    override func backward(withError error: Matrix) -> Matrix {
        dx = inputCache.map(withFunction: sigmoid_deriv).hadamard(other: error)
        return dx
    }

    override func update(withLR lr: Float) {}
}