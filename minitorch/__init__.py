"""A minimalist deep learning library for educational purposes.

This library implements core deep learning components from scratch,
providing a clean, simple implementation for learning purposes.

Submodules
----------
testing : Testing utilities and base classes for unit tests
tensor_data : Core tensor data structures and operations
tensor : Main tensor class and basic operations
tensor_ops : Advanced tensor operations and manipulations
tensor_functions : Mathematical functions for tensor operations
datasets : Dataset loading and processing utilities
optim : Optimization algorithms (e.g., SGD)
module : Base classes for neural network modules
autodiff : Automatic differentiation engine
scalar : Scalar value operations with autograd support
scalar_functions : Mathematical functions for scalar operations

Notes
-----
- All submodules are imported with wildcard (*) for ease of use
- Type checking is ignored for testing imports
- Flake8 warnings about unused imports are suppressed (F401, F403)

"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
