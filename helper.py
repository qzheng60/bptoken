# ---------------------------------
def pair_stats(idxs, counts = None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(idxs, idxs[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pairs(idxs, pair, new_idx):
    """
    Given a list of integers, merge all occurrences of a given pair into a new index
    Example: [1, 2, 3, 1, 2], pair=(1, 2), new_idx=4 -> [4, 3, 4]
    """
    i = 0
    new_idxs = []
    while i < len(idxs):
        if i < len(idxs) - 1 and (idxs[i], idxs[i+1]) == pair:
            new_idxs.append(new_idx)
            i += 2
        else:
            new_idxs.append(idxs[i])
            i += 1
    return new_idxs

# -----------------------------------
def bytepiece_stats(idxs, counts=None):
    counts = {} if counts is None else counts

    for i, idx in enumerate(idxs):
        # Type A: collapse one multibyte piece directly
        if len(idx) > 1:
            key = tuple(idx)
            counts[key] = counts.get(key, 0) + len(idx) - 1

        # Type B: merge two adjacent atomic pieces
        elif (
            i < len(idxs) - 1 
            and len(idx) == 1 
            and len(idxs[i + 1]) == 1
        ):
            key = tuple(idx + idxs[i+1])
            counts[key] = counts.get(key, 0) + 1

    return counts

def merge_bytepieces(idxs, selected, new_idx):
    new_chunk = []
    i = 0

    while i < len(idxs):
        # Case A: collapse one multibyte piece directly
        if len(idxs[i]) > 1 and tuple(idxs[i]) == selected:
            new_chunk.append([new_idx])
            i += 1

        # Case B: merge two adjacent atomic pieces
        elif (
            i < len(idxs) - 1
            and len(idxs[i]) == 1 
            and len(idxs[i + 1]) == 1
            and tuple(idxs[i] + idxs[i + 1]) == selected
        ):
            new_chunk.append([new_idx])
            i += 2

        else:
            new_chunk.append(idxs[i])
            i += 1

    return new_chunk