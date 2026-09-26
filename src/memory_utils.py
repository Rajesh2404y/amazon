import os
import gc
import psutil
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger("entity_resolution")

def get_memory_usage_mb() -> float:
    """Returns current process resident memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_system_available_ram_gb() -> float:
    """Returns system-wide available memory in GB."""
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)

def force_garbage_collection():
    """Forces garbage collection and logs memory reduction."""
    before_mb = get_memory_usage_mb()
    gc.collect()
    after_mb = get_memory_usage_mb()
    logger.debug(f"GC completed: {before_mb:.1f} MB -> {after_mb:.1f} MB (freed {before_mb - after_mb:.1f} MB)")

def log_memory(stage_name: str = ""):
    """Logs current process RSS and system available RAM."""
    rss_mb = get_memory_usage_mb()
    avail_gb = get_system_available_ram_gb()
    logger.info(f"[RAM Check] {stage_name} | Process: {rss_mb:.1f} MB | System Avail: {avail_gb:.2f} GB")

def memory_profiled(func: Callable) -> Callable:
    """Decorator to measure and log memory delta across a function call."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_mem = get_memory_usage_mb()
        logger.info(f"Starting {func.__name__} (RSS: {start_mem:.1f} MB)")
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_mem = get_memory_usage_mb()
            logger.info(f"Finished {func.__name__} (RSS: {end_mem:.1f} MB, Delta: {end_mem - start_mem:+.1f} MB)")
            gc.collect()
    return wrapper
