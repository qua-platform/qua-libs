from qm.qua import wait, align, for_, declare


def safe_wait(t, max_wait=int(1e6)):
    """
    Execute QUA wait commands for the given time t.
    
    Args:
        t: Integer representing the wait time in clock cycles
        max_wait: Maximum wait time per command
        
    Returns:
        None. If t <= 0, does nothing.
        If t <= max_wait, executes wait(t).
        If t > max_wait, executes multiple consecutive wait commands
        that sum to t, with each command <= max_wait.
        
    Raises:
        ValueError: If t is not an integer
    """
    # Check if t is an integer
    if not isinstance(t, int):
        raise ValueError(f"Expected integer, got {type(t).__name__}: {t}")
    
    # If t is 0 or negative, do nothing
    if t <= 0:
        return
    
    # If t is <= max_wait, execute single wait command
    if t <= max_wait:
        wait(t)
        return
    
    # For t > max_wait, break into chunks and execute
    # Calculate how many full chunks we need
    num_full_chunks = t // max_wait
    remainder = t % max_wait
    
    # Execute full chunks using a QUA for_ loop
    i = declare(int)  # Define a QUA variable
    with for_(i, 0, i < num_full_chunks, i + 1):
        wait(max_wait)
    
    # Execute remainder if any
    if remainder > 0:
        wait(remainder)
        
    # reccomend to align after long wait
    if num_full_chunks > 1:
        align()
