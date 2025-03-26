import time
import logging
import functools
from typing import Callable, Dict, Any
from collections import defaultdict
import inspect

# Global dictionary to store timing statistics
timing_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "min_time": float('inf'), "max_time": 0.0})
logger = logging.getLogger(__name__)

def profile(func: Callable) -> Callable:
    """
    Decorator to profile function execution time.
    
    Args:
        func: The function to profile
        
    Returns:
        Wrapped function with profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Update statistics
        func_name = f"{func.__module__}.{func.__name__}"
        timing_stats[func_name]["count"] += 1
        timing_stats[func_name]["total_time"] += elapsed_time
        timing_stats[func_name]["min_time"] = min(timing_stats[func_name]["min_time"], elapsed_time)
        timing_stats[func_name]["max_time"] = max(timing_stats[func_name]["max_time"], elapsed_time)
        
        # Log individual execution time for key functions
        logger.debug(f"PROFILING - {func_name} executed in {elapsed_time:.4f} seconds")
        
        return result
    return wrapper

def print_profiling_stats() -> None:
    """Print a summary of profiling statistics."""
    logger.info("=== Profiling Statistics ===")
    
    # Sort functions by total time (highest first)
    sorted_stats = sorted(
        timing_stats.items(), 
        key=lambda x: x[1]["total_time"], 
        reverse=True
    )
    
    for func_name, stats in sorted_stats:
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        logger.info(f"{func_name}:")
        logger.info(f"  Calls: {stats['count']}")
        logger.info(f"  Total time: {stats['total_time']:.4f} seconds")
        logger.info(f"  Avg time: {avg_time:.4f} seconds")
        logger.info(f"  Min time: {stats['min_time']:.4f} seconds")
        logger.info(f"  Max time: {stats['max_time']:.4f} seconds")
    
    logger.info("===========================")


def log_memory_usage(logger, stage_name):
    """Log current memory usage at various pipeline stages."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)
        
        logger.info(f"Memory usage at {stage_name}: {rss_mb:.2f} MB")
        
        # Force garbage collection if memory usage is high
        if rss_mb > 4000:  # 4GB threshold
            import gc
            gc.collect()
            
            # Log after collection
            memory_info = process.memory_info()
            new_rss_mb = memory_info.rss / (1024 * 1024)
            logger.info(f"Memory after forced collection: {new_rss_mb:.2f} MB (released {rss_mb - new_rss_mb:.2f} MB)")
            
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
    except Exception as e:
        logger.warning(f"Error in memory monitoring: {e}")