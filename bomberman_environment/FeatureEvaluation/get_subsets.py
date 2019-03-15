

def get_subsets_recursively(find_subsets):
    if len(find_subsets) == 1:
        return [[], find_subsets.copy()]

    without_last = get_subsets_recursively(find_subsets[:-1].copy())

    with_last = []

    for subset in without_last:
        _subset = subset.copy()
        _subset.append(find_subsets[-1])
        with_last.append(_subset)

    ret = without_last + with_last

    return ret
