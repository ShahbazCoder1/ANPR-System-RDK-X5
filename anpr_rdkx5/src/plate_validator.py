import re

# Valid 2-letter State & UT Codes in India (including BH for Bharat Series)
VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB", "BH"
}

# Indian License Plate RegEx: State(2) + District(1-2) + Series(1-3) + Number(4)
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

def fix_state_code(char0: str, char1: str) -> tuple[str, str] | None:
    """Try to fix the first 2 characters into a valid Indian state code."""
    code = char0 + char1
    if code in VALID_INDIAN_STATES:
        return char0, char1

    # Positional character mapping fixes
    char0_fixes = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '4': 'A', '6': 'G'}
    char1_fixes = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '4': 'A', '6': 'G',
                   'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}

    c0 = char0_fixes.get(char0, char0)
    c1 = char1_fixes.get(char1, char1)

    if (c0 + char1) in VALID_INDIAN_STATES:
        return c0, char1
    if (char0 + c1) in VALID_INDIAN_STATES:
        return char0, c1
    if (c0 + c1) in VALID_INDIAN_STATES:
        return c0, c1

    # Known state code OCR confusion table
    known_fixes = {
        "MJ": ("M", "H"), "MB": ("M", "H"), "ME": ("M", "H"), "M0": ("M", "H0"), "M8": ("M", "H"), "M7": ("M", "H7"),
        "W8": ("W", "B"), "VB": ("W", "B"), "V8": ("W", "B"), "WE": ("W", "B"), "W0": ("W", "B0"), "IB": ("W", "B"), "1B": ("W", "B"), "UB": ("W", "B"),
        "D1": ("D", "L"), "D0": ("D", "L"), "OL": ("D", "L"),
        "K4": ("K", "A"), "K8": ("K", "A"), "KB": ("K", "A"),
        "Y0": ("U", "P"), "YP": ("U", "P"), "YE": ("U", "P"),
        "T5": ("T", "S"), "7N": ("T", "N"), "7S": ("T", "S"),
        "H8": ("H", "R"), "HB": ("H", "R"),
        "G1": ("G", "J"), "0D": ("O", "D"),
    }
    if code in known_fixes:
        fixed_tuple = known_fixes[code]
        return fixed_tuple[0], fixed_tuple[1]

    return None

def clean_and_validate_plate(raw_text: str) -> str | None:
    """Clean OCR text, auto-correct confusions, validate Indian plate format."""
    if not raw_text:
        return None

    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    if len(cleaned) < 7 or len(cleaned) > 12:
        return None

    fixed = list(cleaned)

    # Fix state code
    state_fix = fix_state_code(fixed[0], fixed[1])
    if state_fix is None:
        return None
    fixed[0], fixed[1] = state_fix

    # Fix district digits (positions 2-3 must be numbers)
    for i in range(2, min(4, len(fixed))):
        ch = fixed[i]
        if not ch.isdigit():
            digit_map = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
            if ch in digit_map:
                fixed[i] = digit_map[ch]

    # Fix last 4 digits
    for i in range(max(4, len(fixed) - 4), len(fixed)):
        ch = fixed[i]
        if not ch.isdigit():
            digit_map = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
            if ch in digit_map:
                fixed[i] = digit_map[ch]

    result = "".join(fixed)

    if result[:2] in VALID_INDIAN_STATES and INDIAN_PLATE_REGEX.match(result):
        return result

    return None
