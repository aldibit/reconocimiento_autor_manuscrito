import os, shutil, random, pathlib

random.seed(42)                        # reproducible split
root = pathlib.Path("dataset")         # where author_001 … author_005 live
out  = pathlib.Path("dataset_split")

for author_dir in root.iterdir():
    images = sorted(author_dir.glob("*.png"))
    random.shuffle(images)

    train, val, test = images[:14], images[14:17], images[17:]

    for subset, group in zip(
        ["train", "val", "test"], [train, val, test]
    ):
        dest = out / subset / author_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in group:
            shutil.copy(img, dest / img.name)
