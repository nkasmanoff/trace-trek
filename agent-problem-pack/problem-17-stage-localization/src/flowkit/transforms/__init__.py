"""Pipeline stage implementations.

Importing this package registers every stage with the core registry. The
mapping from stage name to implementation is defined here and nowhere else.
"""

from src.flowkit.core.registry import register
from src.flowkit.transforms.numeric import cumsum, diff
from src.flowkit.transforms.scaling import clip, minmax_scale
from src.flowkit.transforms.smoothing import exponential_smoothing
from src.flowkit.transforms.windows import rolling_mean

register("smooth")(rolling_mean)
register("ewma")(exponential_smoothing)
register("scale")(minmax_scale)
register("clip")(clip)
register("diff")(diff)
register("cumsum")(cumsum)
