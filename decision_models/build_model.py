"""Build the small serving model from the frozen training split; no test data is read."""

import hashlib
import json
from zipfile import ZipFile

import joblib
from threadpoolctl import threadpool_limits

from .baselines import make
from .schemas import API_MODEL, HERE, annotation


def build_model():
    with ZipFile(HERE / "data/filings-10k-splits.zip") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        content = archive.read("train.jsonl")
    if hashlib.sha256(content).hexdigest() != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Training split checksum mismatch")
    rows = [json.loads(line) for line in content.decode("utf-8").split("\n") if line.strip()]
    model = make("tfidf", class_weight="balanced", max_iter=1000)
    with threadpool_limits(limits=4):
        model.fit([row["body"] for row in rows], [annotation(row)["label"] for row in rows])
    API_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": "tfidf_balanced",
            "model": model,
            # Fixed from the completed validation search; these are not calibration.
            "offsets": {"routine": 0.0, "review_worthy": 0.0, "unclear": -0.75},
            "train_sha256": manifest["splits"]["train"]["sha256"],
        },
        API_MODEL,
    )
    print(f"Built {API_MODEL} from {len(rows)} training documents")


if __name__ == "__main__":
    build_model()
