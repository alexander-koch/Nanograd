
let ds = makeMoons()
let dl = DataLoader(basedOn: ds, withBatchSize: 4)
let dc = DataCollection(trainLoader: dl)

let model = Model(withLayers: [
    Dense(from: 2, to: 16),
    Activation(rows: 1, cols: 16),
    Dense(from: 16, to: 16),
    Activation(rows: 1, cols: 16),
    Dense(from: 16, to: 1),
    Activation(rows: 1, cols: 1)
])

let learner = Learner(onData: dc, withModel: model, withLoss: MSELoss())
learner.fit(epochs: 1000, lr: 1e-6)