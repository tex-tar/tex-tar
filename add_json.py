# push_test_split.py

import json
from pathlib import Path
from datasets import Dataset, Features, Image, Sequence, Value

# 1) Load your labels.json
labels = json.loads(Path("test/testset_labels.json").read_text())

# 2) Build one record per image
records = []
for fname, ann_list in labels.items():
    records.append({
        "image": str(Path("test/textar-testset") / fname),
        "annotations": ann_list
    })

# 3) Define the schema
features = Features({
    "image": Image(decode=True),     # <— decode=True makes it embed the bytes
    "annotations": Sequence({
        "bb_dim": Sequence(Value("int64"), length=4),
        "bb_ids": Sequence({
            "id":   Value("int64"),
            "ocrv": Value("string"),
            "attb": {
                "bold":       Value("bool"),
                "italic":     Value("bool"),
                "b+i":        Value("bool"),
                "no_bi":      Value("bool"),
                "no_us":      Value("bool"),
                "underlined": Value("bool"),
                "strikeout":  Value("bool"),
                "u+s":        Value("bool"),
            }
        })
    })
})

# 4) Create the Dataset
ds = Dataset.from_list(records, features=features)

# 5) Push *just* the test split (replace with your namespace/repo)
ds.push_to_hub("Tex-Tar/MMTAD", split="test")