#!/usr/bin/env python3
"""
Development server entry point for ZaaKy AI Platform Backend

This script starts the FastAPI development server with auto-reload enabled.

Usage:
    python start_server.py

Or via Makefile:
    make dev
"""

import uvicorn

if __name__ == "__main__":
    # Development server with auto-reload
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
