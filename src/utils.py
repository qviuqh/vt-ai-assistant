def generate_session_id():
    """Generates a random session ID."""
    import random
    import string
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=16))