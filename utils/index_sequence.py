import numpy as np

def index_sequence(
    sequence
):
    transform = {}
    set_sequence = set(sequence)
    for index,orig in enumerate(set_sequence):
        transform[orig] = index
    indexed_sequence = np.zeros(len(sequence),dtype=np.int64)
    for i,element in enumerate(sequence):
        indexed_element = transform[element]
        indexed_sequence[i] = indexed_element
    return indexed_sequence

def get_sequence_pairs(
    sequence,
    consider_temporal_order_in_tuples=True, 
):
    '''
        If consider_temporal_order_in_tuples == False, it considers AB and BA the same.
    '''
    sequence_pairs = []
    old = sequence[0]
    for new in sequence[1:]:
        if consider_temporal_order_in_tuples:
            pair = [old,new]
        else:
            pair = sorted([old,new])
        sequence_pairs.append(tuple(pair))
        old = new
    indexed_sequence_pairs = index_sequence(sequence_pairs)
    return sequence_pairs, indexed_sequence_pairs