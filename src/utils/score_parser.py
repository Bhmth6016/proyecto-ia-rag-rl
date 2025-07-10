def parse_binary_score(text: str) -> str:
    text = text.strip().lower()
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    return "unknown"