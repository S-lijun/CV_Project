import json
from pathlib import Path

p = Path(r"c:\Users\lijun\Downloads\CV_Project\data\project.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
changed_cells = []

for i, c in enumerate(nb.get("cells", [])):
    src = "".join(c.get("source", []))
    if "out_png = export_dir /" in src or "YOLO infer" in src:
        original = src

        # ensure yolo_dir init exists near export_dir setup
        anchor = (
            'export_dir = (YOLO_TRAJ_EXPORT_ROOT / SELECTED_TRAJECTORY).resolve()\n'
            'export_dir.mkdir(parents=True, exist_ok=True)\n'
            'plots_dir = export_dir / "Plots"\n'
            'plots_dir.mkdir(parents=True, exist_ok=True)\n'
        )
        if 'yolo_dir = export_dir / "yolo"' not in src and anchor in src:
            src = src.replace(
                anchor,
                anchor +
                'yolo_dir = export_dir / "yolo"\n'
                'yolo_dir.mkdir(parents=True, exist_ok=True)\n\n'
                '# Move old root-level *_yolo.png into yolo/ once\n'
                'for old_png in sorted(export_dir.glob("*_yolo.png")):\n'
                '    target = yolo_dir / old_png.name\n'
                '    if not target.exists():\n'
                '        old_png.replace(target)\n',
                1,
            )

        src = src.replace('out_png = export_dir / f"{rpath.stem}_yolo.png"', 'out_png = yolo_dir / f"{rpath.stem}_yolo.png"')

        if src != original:
            nb["cells"][i]["source"] = [line + "\n" for line in src.splitlines()]
            changed_cells.append(i)

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("changed_cells:", changed_cells)
