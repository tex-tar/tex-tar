import os
import shutil
import yaml
from pathlib import Path

class ImageOrganizer:
  
    def __init__(self, source_folder: str, dest_folder: str):
        self.source = Path(source_folder)
        self.dest   = Path(dest_folder)
        if not self.source.is_dir():
            raise ValueError(f"Source folder does not exist: {self.source}")
        self.dest.mkdir(parents=True, exist_ok=True)

    def _parse_filename(self, filename: str):
       
        parts = filename.split("_CW_", 1)
        if len(parts) != 2:
            return None, None
        cw_id = parts[1].split("_", 1)[0]
        name_parts = parts[0].split("_")[1:]
        image_name = "_".join(name_parts) if name_parts else None
        return image_name, cw_id

    def organize(self):

        for file in self.source.iterdir():
            if not file.is_file() or file.suffix.lower() not in (".png",".jpg",".jpeg"):
                continue
            image_name, cw_id = self._parse_filename(file.name)
            if not image_name or not cw_id:
                continue

            target_dir = self.dest / image_name / cw_id
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, target_dir / file.name)

def main():
    path = "config.yaml"
    with open(path, 'r') as f:
       cfg = yaml.safe_load(f)
    org_cfg = cfg["image_organization"]
    organizer = ImageOrganizer(source_folder = org_cfg["source_folder"],dest_folder = org_cfg["dest_folder"])
    organizer.organize()
    print(f"Organized crops into: {org_cfg['dest_folder']}")

if __name__ == "__main__":
    main()