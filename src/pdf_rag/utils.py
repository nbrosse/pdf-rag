import re


def postprocess_markdown_output(response: str) -> str:
    pattern = r"```markdown\s*(.*?)(```|$)"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return response
    except Exception as e:
        return f"postprocessing markdown failed: {e}"


def postprocess_output(response: str) -> str:
    """
    Extracts the content within a markdown code block, allowing for optional language specification.

    Args:
        response: The string containing the response, potentially with a markdown code block.

    Returns:
        The content within the markdown code block (if found), stripped of leading/trailing whitespace.
        If no code block is found, returns the original response.
        If postprocessing fails, returns an error message.
    """
    pattern = r"```([a-zA-Z]*)?\s*(.*?)(```|$)"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(2).strip()
        else:
            return response
    except Exception as e:
        return f"postprocessing markdown failed: {e}"