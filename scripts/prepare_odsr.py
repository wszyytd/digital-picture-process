"""Validate and convert extracted ODSR-IHS XML to a reproducible YOLO dataset.
Run with the model Python (Pillow required). Never modifies the source.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def yolo_box(box, width, height):
    x1, y1, x2, y2 = box
    if not (width > 0 and height > 0 and all(math.isfinite(v) for v in box)
            and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid box {box} for {width}x{height}")
    # Match the supplied dataset TXT convention: no +/-1 pixel adjustment.
    return ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)

def clean_name(image_id, index, name, box, rules, names):
    fixes = [f for f in rules["fixes"] if f["image_id"] == image_id and f["object_index"] == index]
    if fixes:
        if len(fixes) != 1 or fixes[0]["before"] != name or fixes[0]["bbox"] != box:
            raise ValueError(f"Correction guard failed: {image_id}/{index}")
        name = fixes[0]["after"]
    if name not in names:
        raise ValueError(f"Unreviewed class: {image_id}/{index}/{name}")
    return name

def validate_splits(splits, ids):
    used = set()
    for split, rows in splits.items():
        if len(rows) != len(set(rows)) or used.intersection(rows):
            raise ValueError(f"Duplicate IDs in/across splits: {split}")
        used.update(rows)
    if used != ids:
        raise ValueError("Split coverage differs from source IDs")

def indexed(folder, suffixes):
    result = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes:
            if p.stem in result: raise ValueError(f"Duplicate source ID: {p}")
            result[p.stem] = p
    return result

def main():
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Extracted directory, or a parent containing one Annotations folder")
    parser.add_argument("--output", type=Path, default=ROOT / "data/odsr-ihs")
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "Annotations").is_dir():
        matches = list(source.rglob("Annotations"))
        if len(matches) != 1: raise ValueError("Expected exactly one Annotations directory")
        source = matches[0].parent
    out = args.output.resolve()
    if out.exists(): raise ValueError(f"Refusing existing output: {out}")
    if out == source or source in out.parents: raise ValueError("Output must be outside source")
    config = json.loads((ROOT / "configs/datasets/odsr-ihs.json").read_text(encoding="utf-8"))
    rules_bytes = (ROOT / "configs/datasets/odsr-ihs-cleaning.json").read_bytes()
    rules = json.loads(rules_bytes)
    names = config["names"]
    images = indexed(source / "JPEGImages", {".jpg", ".jpeg", ".png", ".bmp"})
    xmls = indexed(source / "Annotations", {".xml"})
    if set(images) != set(xmls) or len(images) != 6000:
        raise ValueError("Expected 6000 matching image/XML IDs for this cleaning policy")
    splits = {s: (source / "ImageSets/Main" / (s+".txt")).read_text(encoding="utf-8-sig").split() for s in ("train", "val", "test")}
    validate_splits(splits, set(images))
    excluded = set(rules["excluded_images"])
    if not excluded <= set(images): raise ValueError("Exclusion IDs missing")
    assignments = {i:s for s,rows in splits.items() for i in rows}
    records, hashes, counts = [], defaultdict(list), {s:Counter() for s in splits}
    original_objects, corrected, seen_fixes = 0, 0, set()
    for image_id in sorted(images):
        raw = images[image_id].read_bytes()
        with Image.open(images[image_id]) as im:
            im.load()
            width,height = im.size
        xml_data = xmls[image_id].read_bytes()
        tree = ET.fromstring(xml_data)
        if (int(tree.findtext("size/width")),int(tree.findtext("size/height"))) != (width,height):
            raise ValueError(f"XML/image size mismatch: {image_id}")
        objects = tree.findall("object")
        original_objects += len(objects)
        rows, boxes = [], []
        for index,obj in enumerate(objects):
            box = [float(obj.findtext("bndbox/"+k)) for k in ("xmin","ymin","xmax","ymax")]
            coords = yolo_box(box,width,height)
            if obj.findtext("difficult","0") != "0": raise ValueError(f"Unreviewed difficult object: {image_id}")
            if image_id in excluded: continue
            before = obj.findtext("name", "").strip()
            name = clean_name(image_id,index,before,box,rules,names)
            if name != before:
                corrected += 1; seen_fixes.add((image_id,index))
            rows.append(str(names.index(name))+" "+" ".join(f"{v:.10f}" for v in coords))
            boxes.append([name,box])
            counts[assignments[image_id]][name] += 1
        record = {"id":image_id,"split":assignments[image_id],"excluded":image_id in excluded,
                  "image_sha256":digest(raw),"xml_sha256":digest(xml_data),"objects":len(rows)}
        if image_id not in excluded:
            hashes[digest(raw)].append(image_id)
        record["label_text"] = "\n".join(rows)+("\n" if rows else "")
        record["boxes"] = boxes
        records.append(record)
    if seen_fixes != {(f["image_id"],f["object_index"]) for f in rules["fixes"]}:
        raise ValueError("Not all corrections were applied")
    duplicates = [v for v in hashes.values() if len(v)>1]
    cross_split = [v for v in duplicates if len({assignments[i] for i in v})>1]
    if cross_split: raise ValueError(f"Exact duplicate images across splits: {cross_split}")
    # Write only after all source validation has succeeded; no overwrite/resume.
    out.mkdir(parents=True)
    preview_dir = out / "review"
    preview_dir.mkdir()
    preview_ids = {f["image_id"] for f in rules["fixes"]}
    for split,ids in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        preview_ids.update(i for i in ids[:4] if i not in excluded)
    for rec in records:
        image_id,split = rec["id"],rec["split"]
        label_text, boxes = rec.pop("label_text"),rec.pop("boxes")
        if rec["excluded"]: continue
        dest = out / "images" / split / (image_id+images[image_id].suffix.lower())
        shutil.copyfile(images[image_id],dest)
        label_path = out / "labels" / split / (image_id+".txt")
        label_path.write_text(label_text, encoding="utf-8", newline="\n")
        rec["label_sha256"] = digest(label_path.read_bytes())
        if image_id in preview_ids:
            with Image.open(dest) as im:
                im=im.convert("RGB"); draw=ImageDraw.Draw(im)
                for name,box in boxes:
                    draw.rectangle(box,outline="red",width=2)
                    draw.text((box[0],max(0,box[1]-12)),name,fill="red")
                im.save(preview_dir / (image_id+".jpg"))
    summary = {"version":rules["version"],"rules_sha256":digest(json.dumps(rules, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),"names":names,
               "source_images":len(images),"source_objects":original_objects,
               "retained_images":sum(not r["excluded"] for r in records),
               "retained_objects":sum(r["objects"] for r in records),"corrected_objects":corrected,
               "excluded_images":sorted(excluded),"split_images":{s:sum(r["split"]==s and not r["excluded"] for r in records) for s in splits},
               "class_objects":{s:dict(counts[s]) for s in splits},"exact_duplicate_groups":duplicates,
               "limitations":["Original image-ID splits retained; room/sequence leakage and near-duplicates not ruled out.",
               "Coordinate conversion matches original TXT: center=(min+max)/2, extent=max-min; no one-pixel offset.",
               "No person, pet, or robot classes. Not a complete dynamic-SLAM dataset."]}
    save(out / "summary.json",summary)
    save(out / "manifest.json",records)
    save(out / "cleaning.json",rules)
    for s,ids in splits.items():
        (out / (s+".txt")).write_text("\n".join(i for i in ids if i not in excluded)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("Ready:",out)

if __name__ == "__main__":
    main()
