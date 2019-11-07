import sys
sys.path.append('..')
from bias import datasets

sel, embed = datasets.get_test_embedding_set("extreme_biased", "mbfc", "as", 15000, 500, 13, "w2v", "sequence")
