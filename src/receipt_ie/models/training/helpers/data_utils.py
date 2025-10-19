import json
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def sanitize_box(box, W, H):
    xmin, ymin, xmax, ymax = box
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(W - 1, xmax), min(H - 1, ymax)
    return [xmin, ymin, xmax, ymax]

def load_ocr_file(path):
    data = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            try:
                coords = list(map(int, parts[:8]))
            except:
                continue
            text = ",".join(parts[8:]).strip()
            if not text:
                continue
            xmin, ymin = min(coords[::2]), min(coords[1::2])
            xmax, ymax = max(coords[::2]), max(coords[1::2])
            data.append((text, [xmin, ymin, xmax, ymax]))
    return data

def load_entities_txt(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.loads(f.read().strip())
    except Exception:
        return {"company": "", "date": "", "address": "", "total": ""}