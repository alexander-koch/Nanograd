import Foundation
import Accelerate

func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
}

func sigmoidDeriv(_ x: Float) -> Float {
    return sigmoid(x) * (1.0 - sigmoid(x))
}

class Activation: Layer {
    let inputSize: Int
    let outputSize: Int
    var inputCache: Matrix

    init(rows inDim: Int, cols outDim: Int) {
        inputSize = inDim
        outputSize = outDim
        inputCache = Matrix(rows: 1, cols: inputSize)
    }

    func forward(withInput input: Matrix) -> Matrix {
        /*withUnsafeMutablePointer(to: &inputCache.data[0]) { incachepointer in
            withUnsafePointer(to: input.data[0]) { inpointer in
                // inputCache := input
                cblas_scopy(Int32(input.cols), inpointer, 1, incachepointer, 1)
            }
        }*/
        inputCache = input
        return input.map(withFunction: sigmoid)
    }

    func backward(withError error: Matrix) -> Matrix {
        return inputCache.map(withFunction: sigmoidDeriv).hadamard(other: error)
    }

    func update(withLR lr: Float) {}
}