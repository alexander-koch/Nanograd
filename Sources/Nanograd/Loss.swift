
func mse(withPrediction prediction: Matrix, withTarget target: Matrix) -> Float {
    return (prediction - target).square().mean()
}

func mseDeriv(withPrediction prediction: Matrix, withTarget target: Matrix) -> Matrix {
    return (prediction - target).scale(withFactor: 2.0 / Float((prediction.rows * prediction.cols)))
}

protocol Loss {
    func forward(withPrediction prediction: Matrix, withTarget target: Matrix) -> Float
    func backward(withPrediction prediction: Matrix, withTarget target: Matrix) -> Matrix
}

struct MSELoss: Loss {
    func forward(withPrediction prediction: Matrix, withTarget target: Matrix) -> Float {
        return mse(withPrediction: prediction, withTarget: target)
    }

    func backward(withPrediction prediction: Matrix, withTarget target: Matrix) -> Matrix {
        return mseDeriv(withPrediction: prediction, withTarget: target)
    }
}