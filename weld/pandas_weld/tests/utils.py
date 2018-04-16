from lazy_result import LazyResult


def evaluate_if_necessary(data):
    if isinstance(data, LazyResult):
        data = data.evaluate(verbose=False)

    return data


def evaluate_array_if_necessary(array):
    for i in xrange(len(array)):
        array[i] = evaluate_if_necessary(array[i])

    return array
