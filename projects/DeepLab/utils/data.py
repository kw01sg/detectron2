import logging
from pathlib import Path

IGNORE_LABEL = -1


def get_synthia_dataset(input_path: Path, label_path: Path):
    HEIGHT, WIDTH = 760, 1280
    image_dir = input_path / "RAND_CITYSCAPES/RGB"

    dict_list = []
    for image_path in image_dir.glob("*.png"):
        image_name = image_path.name
        annotation = {}

        annotation['file_name'] = str(image_path.resolve())
        annotation['height'] = HEIGHT
        annotation['width'] = WIDTH
        annotation['sem_seg_file_name'] = str((label_path / image_name).resolve())

        dict_list.append(annotation)

    logging.info(f"Loading SYNTHIA dataset of size: {len(dict_list)}")

    return dict_list
