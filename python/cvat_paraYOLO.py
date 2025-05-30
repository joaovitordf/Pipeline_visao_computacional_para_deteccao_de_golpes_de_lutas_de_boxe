import os
from xml.dom import minidom

# --- CONFIGURAÇÃO ---
XML_PATH = "annotations.xml"
IMG_DIR = "treinamento/images"
OUT_DIR = "treinamento/labels"

CLASS_MAP = {
    "vermelho": 0,
    "azul": 1
}

os.makedirs(OUT_DIR, exist_ok=True)
doc = minidom.parse(XML_PATH)

for image in doc.getElementsByTagName('image'):
    img_name = image.getAttribute('name')
    w = float(image.getAttribute('width'))
    h = float(image.getAttribute('height'))
    base = os.path.splitext(img_name)[0]

    lines = []

    for box in image.getElementsByTagName('box'):
        label = box.getAttribute('label')
        cls = CLASS_MAP.get(label, None)
        if cls is None:
            print(f"Aviso: classe '{label}' não mapeada em CLASS_MAP — pulando.")
            continue

        xtl = float(box.getAttribute('xtl'))
        ytl = float(box.getAttribute('ytl'))
        xbr = float(box.getAttribute('xbr'))
        ybr = float(box.getAttribute('ybr'))

        xc = (xtl + (xbr - xtl) / 2) / w
        yc = (ytl + (ybr - ytl) / 2) / h
        bw = (xbr - xtl) / w
        bh = (ybr - ytl) / h

        pts_elem = None
        for pts in image.getElementsByTagName('points'):
            if pts.getAttribute('label') == label:
                pts_elem = pts
                break
        if pts_elem is None:
            print(f"Aviso: pontos não encontrados para label '{label}' em {img_name}.")
            continue

        raw_pts = pts_elem.getAttribute('points').split(';')
        norm_pts = []
        for p in raw_pts:
            x, y = map(float, p.split(','))
            norm_pts += [x / w, y / h]

        vals = [str(cls),
                f"{xc:.6f}", f"{yc:.6f}",
                f"{bw:.6f}", f"{bh:.6f}"] + [f"{v:.6f}" for v in norm_pts]
        lines.append(" ".join(vals))

    if lines:
        with open(os.path.join(OUT_DIR, f"{base}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
