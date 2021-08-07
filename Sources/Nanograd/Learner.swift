
func mse(withPrediction prediction: Matrix, withTarget target: Matrix) -> Float {
    return (prediction - target).square().mean()
}

func mse_deriv(withPrediction prediction: Matrix, withTarget target: Matrix) -> Matrix {
    return (prediction - target).scale(withFactor: 2.0 / Float((prediction.rows * prediction.cols)))
}

protocol Loss {
    func forward(withPrediction prediction: Matrix, withTarget target: Matrix) -> Float
    func backward(withPrediction prediction: Matrix, withTarget target: Matrix) -> Matrix
}

struct MSELoss {

}

class Learner {
    var dataCollection: DataCollection
    var model: Model
    var loss: Loss

    init(onData d: DataCollection, withModel m: Model, withLoss l: Loss) {
        dataCollection = d
        model = m
        loss = l
    }

    func fit(epochs: Int, lr: Float) {
        for epoch in 0..<epochs {
            for i in 0..<dataCollection.trainLoader.length() {
                let batch = dataCollection.trainLoader[i]
                let batchScale = Float(batch.count)
                var totalLoss: Float = 0.0
                for (X,y) in batch {
                    let yHat = model.forward(withInput: X)
                    totalLoss += loss.forward(withPrediction: yHat, withTarget: y) * batchScale
                    let error = loss.backward(withPrediction: yHat, withTarget: y)
                    model.backward(withError: error)
                }
                model.update(withLR: lr)
                print("Loss: \(totalLoss)")
            }
        }
    }
}