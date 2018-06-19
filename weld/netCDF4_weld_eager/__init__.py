from dataset import Dataset
from variable import Variable


""" The only difference between this eager and the lazy parser is a single line in Variable.lazy_slice_rows 
which used to remember the slice. Ideally, the eager implementation should be easier as this uses the 
lazy-parsing framework when it shouldn't. For the time being, can't find an effective solution to handle it properly,
as exemplified in LazyResult.update_rows """
