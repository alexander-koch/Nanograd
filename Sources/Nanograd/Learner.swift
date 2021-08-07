
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
                    //print("yhat: \(yHat), \(y)")
                    totalLoss += loss.forward(withPrediction: yHat, withTarget: y) * batchScale
                    //print(totalLoss)
                    let error = loss.backward(withPrediction: yHat, withTarget: y)
                    //print(error)
                    model.backward(withError: error)
                    //print("backward")
                }
                model.update(withLR: lr)
                print("\repoch: \(epoch) loss: \(totalLoss)", terminator:"")
            }
        }
    }
}