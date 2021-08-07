
class Layer {
    func forward(withInput input: Matrix) -> Matrix {
        return input
    }

    /*func forward(withBatch batch: Array<Matrix>) -> Array<Matrix> {
        var out: Array<Matrix> = []
        out.reserveCapacity(batch.count)
        for data in batch {
            out.append(forward(withInput: data))
        }
        return out
    }*/

    func backward(withError error: Matrix) -> Matrix {
        return error
    }

    /*func backward(withBatch batch: Array<Matrix>) -> Array<Matrix> {
        var out: Array<Matrix> = []
        out.reserveCapacity(batch.count)
        for data in batch {
            out.append(backward(withError: data))
        }
        return out
    }*/

    func update(withLR lr: Float) {}
}
