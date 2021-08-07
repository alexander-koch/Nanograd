class Matrix: CustomStringConvertible {
    var rows: Int
    var cols: Int
    var data: Array<Float>

    init(rows r: Int, cols c: Int) {
        rows = r
        cols = c
        data = [Float](repeating: 0, count: r * c)
    }

    class func eye(size: Int) -> Matrix {
        let matrix = Matrix(rows: size, cols: size)
        for j in 0..<size {
            for i in 0..<size {
                if i == j {
                    matrix.data[i + j * size] = 1
                } else {
                    matrix.data[i + j * size] = 0
                }
            }
        }
        return matrix
    }

    class func zeros_like(reference: Matrix) -> Matrix {
        return Matrix(rows: reference.rows, cols: reference.cols)
    }

    func clear() {
        for i in 0..<rows*cols {
            data[i] = 0
        }
    }

    func mean() -> Float {
        var mean: Float = 0.0
        let scale = 1.0 / Float(rows * cols)
        for i in 0..<rows*cols {
            mean += data[i] * scale
        }
        return mean
    }

    func square() -> Matrix {
        let matrix = Matrix(rows: rows, cols: cols)
        for i in 0..<rows*cols {
            matrix.data[i] = data[i] * data[i]
        }
        return matrix
    }

    func scale(withFactor f: Float) -> Matrix {
        let scaledMatrix = Matrix(rows: rows, cols: cols)
        for i in 0..<rows*cols {
            scaledMatrix.data[i] *= f
        }
        return scaledMatrix
    }

    func scale_(withFactor f: Float) {
        for i in 0..<rows*cols {
            data[i] *= f
        }
    }

    class func uniform(rows r: Int, cols c: Int) -> Matrix {
        let matrix = Matrix(rows: r, cols: c)
        for i in 0..<r*c {
            matrix.data[i] = Float.random(in: -1...1)
        }
        return matrix
    }

    func map(withFunction fn: (Float) -> Float) -> Matrix {
        let matrix = Matrix(rows: rows, cols: cols)
        for i in 0..<rows*cols {
            matrix.data[i] = fn(data[i])
        }
        return matrix
    }

    func map_(withFunction fn: (Float) -> Float) {
        for i in 0..<rows*cols {
            data[i] = fn(data[i])
        }
    }

    func hadamard(other: Matrix) -> Matrix {
        let matrix = Matrix(rows: rows, cols: cols)
        for i in 0..<rows*cols {
            matrix.data[i] = data[i] * other.data[i]
        }
        return matrix
    }

    class func asRowVec(data: Array<Float>) -> Matrix {
        let matrix = Matrix(rows: 1, cols: data.count)
        matrix.data = data
        return matrix
    }

    class func asColVec(data: Array<Float>) -> Matrix {
        let matrix = Matrix(rows: data.count, cols: 1)
        matrix.data = data
        return matrix
    }

    subscript(row: Int, column: Int) -> Float {
        return data[column + row * cols]
    }

    var description: String {
        return "\(data)"
    }
}

extension Matrix {
    static func -(left: Matrix, right: Matrix) -> Matrix {
        var matrix = Matrix.zeros_like(reference: left)
        for i in 0..<left.rows*left.cols {
            matrix.data[i] = left.data[i] - right.data[i]
        }
        return matrix
    }
}