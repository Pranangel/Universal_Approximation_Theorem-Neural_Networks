#Author: Pranangel
#Purpose: Defining a template for classes in losses.py and errors.py to use.

from typing import Callable
from abc import abstractmethod

class Function:
    """Abstract class housing relevant accessors for a function. This class defines behavior that
    Python function Callables get returned by getFunc() and getDeriv()."""

    @staticmethod
    @abstractmethod
    def getFunc() -> Callable:
        pass
    
    @staticmethod
    @abstractmethod
    def getDeriv() -> Callable:
        pass
