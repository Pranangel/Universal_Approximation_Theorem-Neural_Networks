from typing import Callable
from abc import abstractmethod

class Function:
    """Abstract class housing relevant accessors for a function. Python functions will be
    returned by getFunc() and getDeriv(), not what the functions output."""

    @staticmethod
    @abstractmethod
    def getFunc() -> Callable:
        pass
    
    @staticmethod
    @abstractmethod
    def getDeriv() -> Callable:
        pass