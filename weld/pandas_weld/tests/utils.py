from lazy_result import LazyResult


def evaluate_if_necessary(data):
    if isinstance(data, LazyResult):
        # e.g. for Series this will return a Series with raw data
        data = data.evaluate()
        # so want the np.ndarray within
        if isinstance(data, LazyResult):
            data = data.expr

    return data


def evaluate_array_if_necessary(array):
    for i in xrange(len(array)):
        array[i] = evaluate_if_necessary(array[i])

    return array
