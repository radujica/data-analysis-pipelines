from lazy_data import LazyData


def evaluate_if_necessary(data):
    if isinstance(data, LazyData):
        data = data.evaluate(verbose=False)

    return data


def evaluate_array_if_necessary(array):
    for i in xrange(len(array)):
        array[i] = evaluate_if_necessary(array[i])

    return array
