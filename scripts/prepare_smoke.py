"""Download official COCO8. These eight images are for pipeline verification only."""
from workspace import ROOT, download, extract_zip, save_json

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"


def main():
    archive = ROOT / ".cache/downloads/coco8.zip"
    record = download(URL, archive)
    target = ROOT / "data/coco8"
    if not target.exists():
        extract_zip(archive, ROOT / "data")
    for split in ("train", "val"):
        images = sorted((target / "images" / split).glob("*.jpg"))
        if len(images) != 4:
            raise ValueError("Expected four COCO8 images in " + split)
        for image in images:
            if not (target / "labels" / split / (image.stem + ".txt")).exists():
                raise ValueError("Missing smoke label: " + image.name)
    save_json(target / "source.json", record)
    print("COCO8 ready:", target)
    print("SMOKE TEST ONLY: do not use these metrics as research results.")


if __name__ == "__main__":
    main()
