import logging
import time
from typing import Optional, Any, List
from datetime import datetime, timedelta

class ProgressTracker:
    """Tracks and logs progress of long-running operations."""
    
    def __init__(self, 
                 name: str, 
                 total_steps: int, 
                 log_interval: int = 100, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize progress tracker.
        
        Args:
            name: Name of the operation being tracked
            total_steps: Total number of steps to complete
            log_interval: How often to log progress (in steps)
            logger: Logger instance (optional)
        """
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.completed = False
        
        # Log initial state
        self.logger.info(f"Starting {self.name}: 0/{self.total_steps} (0.00%)")
    
    def update(self, steps: int = 1, current_item: Any = None) -> None:
        """
        Update progress by specified number of steps.
        
        Args:
            steps: Number of steps completed
            current_item: Current item being processed (optional)
        """
        self.current_step += steps
        
        # Check if we should log progress
        if (self.current_step % self.log_interval == 0 or 
            self.current_step == self.total_steps or 
            time.time() - self.last_log_time > 30):  # Also log if 30 seconds have passed
            
            # Calculate percentage and elapsed time
            percentage = (self.current_step / self.total_steps) * 100
            elapsed = time.time() - self.start_time
            
            # Estimate remaining time
            if self.current_step > 0:
                steps_per_second = self.current_step / elapsed
                remaining_steps = self.total_steps - self.current_step
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "Unknown"
            
            # Create log message
            msg = f"Progress {self.name}: {self.current_step}/{self.total_steps} ({percentage:.2f}%)"
            msg += f" - Elapsed: {str(timedelta(seconds=int(elapsed)))}, ETA: {eta}"
            
            if current_item:
                msg += f" - Current: {current_item}"
            
            # Log progress
            self.logger.info(msg)
            self.last_log_time = time.time()
    
    def complete(self) -> None:
        """Mark the task as complete and log final statistics."""
        if not self.completed:
            self.current_step = self.total_steps
            elapsed = time.time() - self.start_time
            
            self.logger.info(
                f"Completed {self.name} in {str(timedelta(seconds=int(elapsed)))} "
                f"({self.total_steps} steps, {self.total_steps / elapsed:.2f} steps/second)"
            )
            self.completed = True