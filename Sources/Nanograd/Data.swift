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
            let leftover = dataset.length() - start
            return dataset[start..<leftover]
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
}