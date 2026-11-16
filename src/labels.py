"""Display names and colors for the six trained vehicle classes."""

CLASS_NAMES_TR = {
    "bus": "Otobüs",
    "car": "Otomobil",
    "microbus": "Minibüs",
    "motorbike": "Motosiklet",
    "pickup-van": "Kamyonet",
    "truck": "Kamyon",
}

CLASS_COLORS = {
    "bus": (37, 99, 235),
    "car": (15, 118, 110),
    "microbus": (217, 119, 6),
    "motorbike": (124, 58, 237),
    "pickup-van": (22, 163, 74),
    "truck": (220, 38, 38),
}

DIRECTION_NAMES_TR = {
    "down": "Aşağı",
    "up": "Yukarı",
    "right": "Sağa",
    "left": "Sola",
}


def class_label(class_name: str) -> str:
    return CLASS_NAMES_TR.get(class_name, class_name)


def class_color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, (71, 85, 105))
