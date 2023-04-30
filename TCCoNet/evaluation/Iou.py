import numpy


def iou(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        iou = tp / float(tp + fp + fn)
    except ZeroDivisionError:
        iou = 0.0

    return iou
