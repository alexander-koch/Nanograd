
protocol Layer {
    func forward(withInput input: Matrix) -> Matrix
    func backward(withError error: Matrix) -> Matrix
    func update(withLR lr: Float)
}
