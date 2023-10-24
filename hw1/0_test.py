import sys

sys.path.append("./python")
sys.path.append("./apps")
from apps.simple_ml import *
import numdifftools as nd

import numpy as np
import mugrade
import python.needle as ndl

t1 = ndl.Tensor([1, 2, 3], dtype="float32")
