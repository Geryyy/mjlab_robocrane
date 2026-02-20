class RateLimiter:
    def __init__(self, sample_time: float, max_rate: float):
        """
        Initialize the rate limiter.
        
        Args:
            sample_time (float): Time between updates (in seconds).
            max_rate (float): Maximum allowed rate of change per second.
        """
        self.sample_time = sample_time
        self.max_delta = max_rate * sample_time
        self.last_value = None

    def update(self, new_value: float) -> float:
        """
        Update the value with rate limiting.
        
        Args:
            new_value (float): The desired new input value.
        
        Returns:
            float: The updated (possibly rate-limited) value.
        """
        if self.last_value is None:
            self.last_value = new_value
            return new_value

        delta = new_value - self.last_value
        if abs(delta) > self.max_delta:
            delta = self.max_delta if delta > 0 else -self.max_delta

        self.last_value += delta
        return self.last_value
    
    def reset(self, reset_value: float = 0):
        """
        Reset the rate limiter to its initial state.
        """
        self.last_value = reset_value

    def get_value(self):
        """
        Get the current value of the rate limiter.
        
        Returns:
            float: The current value.
        """
        return self.last_value