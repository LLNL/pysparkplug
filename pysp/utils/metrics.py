import numpy as np


def classify(data, model, labels=None):

    cnt = len(data)
    data_labels = [u[0] for u in data]

    if labels is None:
        labels = sorted(set(data_labels))

    class_ll = np.zeros((len(data), len(labels)))
    ulabels, true_labels = np.unique(data_labels, return_inverse=True)

    other_labs = sorted(set(labels).difference(list(ulabels)))
    ulabel_map = dict(zip(list(ulabels) + other_labs, range(len(ulabels) + len(other_labs))))

    for label in labels:
        idx = ulabel_map[label]
        loc_data = [(label, u[1]) for u in data]
        class_ll[:,idx] = model.seq_log_density(model.seq_encode(loc_data))

    max_ll = class_ll.max(axis=1, keepdims=True)
    class_ll -= max_ll
    np.exp(class_ll, out=class_ll)
    class_ll /= class_ll.sum(axis=1, keepdims=True)

    class_prob = class_ll[np.arange(cnt), true_labels]
    class_diff = class_ll - class_prob[:,None]
    class_rank = (class_diff >= 0).sum(axis=1) - 1
    data_labels = np.asarray(data_labels)
    class_ll = {label: class_ll[:, ulabel_map[label]] for label in labels}

    return class_rank, class_prob, data_labels, class_ll


def roc_curve(pos_x, neg_x):

    res = np.zeros((len(pos_x)+len(neg_x), 2))
    res[:len(pos_x), 0] = np.asarray(pos_x)
    res[:len(pos_x), 1] = 1
    res[len(pos_x):, 0] = np.asarray(neg_x)
    res[len(pos_x):, 1] = 0

    sidx = np.argsort(-res[:,0])
    res = res[sidx, :]

    pd = np.cumsum(res[:, 1])
    fa = np.cumsum(1-res[:, 1])

    pd /= float(len(pos_x))
    fa /= float(len(neg_x))

    return pd, fa

def roc_percentiles(pos_x, neg_x, perc_points):

    pd, fa = roc_curve(pos_x, neg_x)
    rv = []

    for i in range(len(perc_points)):

        points = (pd <= perc_points[i])

        if np.sum(points) == 0:
            continue

        y = np.max(pd[points])
        x = np.max(fa[pd == y])
        rv.append([x, y])

    return np.asarray(rv)

def ranking_depth(x, k=None, comp_func = lambda a,b: a == b):
    """

    :param x: A list of the form [(search label, [(test label, score)])]
    :param k:
    :return: If k is None, then a list of lists
    """

    if k is not None:
        retval = np.zeros((len(x), k))
        retval.fill(np.nan)
    else:
        retval = []

    idx = 0
    for entry in x:

        scores = np.asarray([u[1] for u in entry[1]])
        matches = np.asarray([comp_func(entry[0], u[0]) for u in entry[1]])

        sidx = np.argsort(-scores)

        matches = matches[sidx]
        scores = scores[sidx]

        ranks = np.arange(len(sidx))[matches]

        if k is not None:
            sz = min(k, len(ranks))
            retval[idx, :sz] = ranks[:sz]
        else:
            retval.append(ranks)

        idx += 1

    return retval


