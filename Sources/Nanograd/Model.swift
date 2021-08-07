
class Model {
    var layers: Array<Layer>
    init(withLayers l: Array<Layer>) {
        layers = l
    }

    func forward(withInput data: Matrix) -> Matrix {
        var currentData = data
        for layer in layers {
            currentData = layer.forward(withInput: currentData)
        }
        return currentData
    }

    func backward(withError error: Matrix) -> Matrix {
        var currentError = error
        for i in stride(from: layers.count - 1, through: 0, by: -1) {
            let last = currentError.shape()
            currentError = layers[i].backward(withError: currentError)
        }
        return currentError
    }

    func update(withLR lr: Float) {
        for layer in layers {
            layer.update(withLR: lr)
        }
    }
}
