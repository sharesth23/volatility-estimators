import ruptures as rpt


def detect_structural_breaks(series, penalty=10):
    algo = rpt.Pelt(model="rbf").fit(series.values)
    return algo.predict(pen=penalty)
