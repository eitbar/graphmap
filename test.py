import numpy as np
from graph_utils import word_graph
import embedding_utils

words = [
    [1,2],
    [2,1],
    [2,2],
    [1,-2],
    [2,-1],
    [2,-2],
]

we = np.array(words, dtype='float32')
embedding_utils.normalize(we, ['unit'])
print(we)
wg = word_graph(we, 0, 2)

print(wg.get_clique_center_lists())
