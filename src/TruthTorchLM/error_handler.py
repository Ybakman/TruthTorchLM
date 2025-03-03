def handle_logprobs_error(func):
    """A decorator that catches 'logprobs' errors and returns
    a custom message or behavior."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError) as e:
            # Check if 'logprobs' is involved in the error
            if "logprobs" in str(e):
                # Here you decide what to do instead of crashing
                # e.g. return a custom string, raise a custom exception, log an error, etc.
                raise ValueError(
                    "This API model does not provide logprobs and cannot be used with truth methods that require logprobs. Please try a different API model or different truth method."
                )

            raise e

    return wrapper
