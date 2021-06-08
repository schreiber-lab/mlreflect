import functools
from timeit import default_timer


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if 'time' in locals() and time is True:
            start_time = default_timer()  # 1
            value = func(*args, **kwargs)
            end_time = default_timer()  # 2
            run_time = end_time - start_time  # 3
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
            return value
        else:
            return func(*args, **kwargs)

    return wrapper_timer
