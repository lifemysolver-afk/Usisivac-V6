import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from pathlib import Path

BASE = Path("/home/ubuntu/Usisivac-V6")
# We don't have the OOF arrays in memory, but we can re-run a simple mean blend 
# or use the best individual model if we had saved them.
# Since we killed the process, let's write a script that does a simple 
# weighted average of the top models if we can recover them, 
# or just re-run the final part of the pipeline with fewer iterations.

print("Fast Blend Script Started...")
# Actually, the best way is to just re-run the final part of the script 
# but with a much smaller number of iterations for weight search.
