import Accelerate

class Dense: Layer {
    let inputSize: Int
    let outputSize: Int
    var weights: Matrix
    var bias: Matrix

    var dw: Matrix
    var db: Matrix
    var dx: Matrix
    var batchSize: Int

    var inputCache: Matrix
    var outputCache: Matrix

    init(from: Int, to: Int) {
        inputSize = from
        outputSize = to
        weights = Matrix.uniform(rows: inputSize, cols: outputSize)
        bias = Matrix(rows: 1, cols: outputSize)
        dw = Matrix(rows: inputSize, cols: outputSize)
        db = Matrix(rows: 1, cols: outputSize)
        dx = Matrix(rows: 1, cols: inputSize)
        batchSize = 0

        weights.scale_(withFactor: (2 / Float(inputSize)).squareRoot())

        inputCache = Matrix(rows: 1, cols: inputSize)
        outputCache = Matrix(rows: 1, cols: outputSize)
    }

    func forward(withInput input: Matrix) -> Matrix {
        /*withUnsafeMutablePointer(to: &inputCache.data[0]) { incachepointer in
            withUnsafePointer(to: input.data[0]) { inpointer in
                // inputCache := input
                cblas_scopy(Int32(input.cols), inpointer, 1, incachepointer, 1)
            }
        }*/
        inputCache = input
        
        withUnsafeMutablePointer(to: &outputCache.data[0]) { outpointer in
            // outputCache := bias
            withUnsafePointer(to: bias.data[0]) { biaspointer in
                cblas_scopy(Int32(bias.cols), biaspointer, 1, outpointer, 1)
            }

            let m = Int32(input.rows)
            let n = Int32(weights.cols)
            let k = Int32(input.cols)

            withUnsafePointer(to: input.data[0]) { inputpointer in
                withUnsafePointer(to: &weights.data[0]) { weightspointer in
                    // outputCache := input * weights + outputCache
                    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        m,n,k,
                        1,inputpointer,k,
                        weightspointer,n,
                        1,outpointer,n)
                }
            }
        }
        return outputCache
    }

    func backward(withError error: Matrix) -> Matrix {
        withUnsafePointer(to: error.data[0]) { errpointer in
            // db := db + error
            withUnsafeMutablePointer(to: &db.data[0]) { dbpointer in
                cblas_saxpy(Int32(error.cols), 1, errpointer, 1, dbpointer, 1)
            }
            
            var m = Int32(inputCache.cols)
            var n = Int32(error.cols)
            var k = Int32(inputCache.rows)

            withUnsafeMutablePointer(to: &dw.data[0]) { dwpointer in
                withUnsafePointer(to: inputCache.data[0]) { incachepointer in
                    // dE / dw = dE / dz * dz / dw = dE / dz * x
                    // dw := dw + inputCache^T * error
                    cblas_sgemm(CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        m,n,k,1,
                        incachepointer,Int32(inputCache.cols),
                        errpointer,n,
                        1,dwpointer,n)
                }
            }
            
            m = Int32(error.rows)
            n = Int32(weights.rows)
            k = Int32(error.cols)
            withUnsafeMutablePointer(to: &dx.data[0]) { dxpointer in
                withUnsafePointer(to: weights.data[0]) { weightspointer in
                    // dE / dx
                    // dx := error * weights^T
                    cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        m,n,k,1,
                        errpointer,k,
                        weightspointer,Int32(weights.cols),
                        0,dxpointer,n)
                }
            }
        }
        batchSize += 1
        return dx
    }

    func update(withLR lr: Float) {
        let scale: Float = 1.0 / Float(batchSize)

        // dw := dw * scale
        let dwlen = Int32(dw.rows * dw.cols)
        withUnsafeMutablePointer(to: &dw.data[0]) { dwpointer in
            cblas_sscal(dwlen, scale, dwpointer, 1)
        }
        // db := db * scale
        let dblen = Int32(db.rows * db.cols)
        withUnsafeMutablePointer(to: &db.data[0]) { dbpointer in
            cblas_sscal(dblen, scale, dbpointer, 1)
        }

        // weights := weights - alpha * dw
        withUnsafeMutablePointer(to: &weights.data[0]) { weightspointer in
            withUnsafePointer(to: dw.data[0]) { dwpointer in
                cblas_saxpy(Int32(dw.rows * dw.cols), -lr, dwpointer, 1, weightspointer, 1)
            }
        }

        // bias := bias - alpha * db
        withUnsafeMutablePointer(to: &bias.data[0]) { biaspointer in
            withUnsafePointer(to: db.data[0]) { dbpointer in
                cblas_saxpy(Int32(db.rows * db.cols), -lr, dbpointer, 1, biaspointer, 1)
            }
        }

        dw.clear()
        db.clear()
        batchSize = 0
    }
}