import json
from datasets import Dataset, Features, Image, Sequence, Value
from pathlib import Path

labels_path = "/projects/data/vision-team/swaroopa_jinka/TEXTAR/MMTAD/test/testset_labels.json"
with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)
    # labels: { "ncert-page_25.png": [ { bb_dim: [...], bb_ids: [...] }, … ], … }


records = []
for fname, ann_list in labels.items():
    records.append({
        "image": str(Path("/projects/data/vision-team/swaroopa_jinka/TEXTAR/MMTAD/test/textar-testset") / fname),
        "annotations": ann_list
    })

# 3) Define the schema so Data Studio knows how to render each column
features = Features({
    "image": Image(),                        # image preview
    "annotations": Sequence({                # a list of dicts
        "bb_dim": Sequence(Value("int64"), length=4),
        "bb_ids": Sequence({                 # sequence of word‐level dicts
            "id":   Value("int64"),
            "ocrv": Value("string"),
            "attb": {                        # nested attributes
                "bold":      Value("bool"),
                "italic":    Value("bool"),
                "b+i":       Value("bool"),
                "no_bi":     Value("bool"),
                "no_us":     Value("bool"),
                "underlined":Value("bool"),
                "strikeout": Value("bool"),
                "u+s":       Value("bool"),
            }
        })
    })
})

# 4) Create the Dataset
ds = Dataset.from_list(records, features=features)

# 5) Push to the Hub (requires login)
ds.push_to_hub("Tex-TAR/MMTAD")