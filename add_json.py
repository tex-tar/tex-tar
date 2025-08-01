import json, pathlib

src_json   = pathlib.Path("/projects/data/vision-team/swaroopa_jinka/TEXTAR/MMTAD/test/testset_labels.json")
out_jsonl  = pathlib.Path("/projects/data/vision-team/swaroopa_jinka/TEXTAR/MMTAD/test/textar_testlabels.jsonl")

with src_json.open() as f:
    mapping = json.load(f)

with out_jsonl.open("w") as out:
    for img, ann in mapping.items():
        out.write(json.dumps({"image": img, "annotation": ann}) + "\n")