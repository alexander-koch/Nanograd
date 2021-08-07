import Foundation

class Dataset {
    var data: Array<Matrix>
    var labels: Array<Matrix>

    init(data d: Array<Matrix>, labels l: Array<Matrix>) {
        data = d
        labels = l
    }

    subscript(index: Int) -> (Matrix, Matrix) {
        return (data[index], labels[index])
    }

    subscript(index: Range<Int>) -> Array<(Matrix, Matrix)> {
        return Array(zip(data[index], labels[index]))
    }

    func length() -> Int {
        return data.count
    }
}

func linspace(start: Float, stop: Float, num: Int) -> Array<Float> {
    if num == 1 {
        return [stop]
    } else {
        let step = (stop - start) / Float(num - 1)
        var out: Array<Float> = []
        out.reserveCapacity(num)
        for i in 0..<num {
            out.append(start + step * Float(i))
        }
        return out
    }
}

func makeMoons(numberOfSamples nSamples: Int = 100) -> Dataset {
    let nSamplesOut = nSamples / 2
    let nSamplesIn = nSamples - nSamplesOut

    let outerCircX = linspace(start: 0, stop: Float.pi, num: nSamplesOut).map { cos($0) }
    let outerCircY = linspace(start: 0, stop: Float.pi, num: nSamplesOut).map { sin($0) }

    let innerCircX = linspace(start: 0, stop: Float.pi, num: nSamplesIn).map { 1 - cos($0) }
    let innerCircY = linspace(start: 0, stop: Float.pi, num: nSamplesIn).map { 1 - sin($0) - 0.5 }

    let xData = outerCircX + innerCircX + outerCircY + innerCircY
    let X = Matrix(rows: 2, cols: nSamples, data: xData).transpose().getRows()
   
    let yData = [Float](repeating: 0, count: nSamplesOut) + [Float](repeating: 1, count: nSamplesIn)
    let y = yData.map({ Matrix(rows: 1, cols: 1, data: [$0 * 2 - 1])})

    return Dataset(data: X, labels: y)
}

class DataLoader {
    var dataset: Dataset
    var batchSize: Int

    init(basedOn data: Dataset, withBatchSize bs: Int) {
        dataset = data
        batchSize = bs
    }

    subscript(index: Int) -> Array<(Matrix, Matrix)> {
        let start = index * batchSize
        let end = start + batchSize
        if end < dataset.length() {
            return dataset[start..<end]
        } else {
            return dataset[start..<dataset.length()]
        }
    }

    func oneBatch() -> Array<(Matrix, Matrix)> {
        return self[0]
    }

    func length() -> Int {
        return Int(ceil(Double(dataset.length()) / Double(batchSize)))
    }
}

struct DataCollection {
    var trainLoader: DataLoader
    var valLoader: DataLoader?
    var testLoader: DataLoader?

    init(trainLoader traindl: DataLoader, valLoader valdl: DataLoader? = nil, testLoader testdl: DataLoader? = nil) {
        trainLoader = traindl
        valLoader = valdl
        testLoader = testdl
    }
}