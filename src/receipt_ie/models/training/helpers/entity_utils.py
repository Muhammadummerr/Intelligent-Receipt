import re

def text_match(token, value):
    """Robust match: handles partial tokens, punctuation, etc."""
    token = token.lower().strip()
    value = value.lower().strip()
    if not token or not value:
        return False
    # split both sides on punctuation/spaces
    parts = re.split(r"[\s,./:\-]", value)
    return token in parts or token in value
