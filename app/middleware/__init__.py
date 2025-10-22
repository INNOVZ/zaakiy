"""
Middleware package for custom request/response processing
"""
from .cors import SmartCORSMiddleware

__all__ = ["SmartCORSMiddleware"]
