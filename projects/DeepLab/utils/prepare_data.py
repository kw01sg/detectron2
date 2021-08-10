import numpy as np
from numpy.core.defchararray import encode
import cv2
from tqdm import tqdm

import click
import logging
from pathlib import Path
from utils.data import IGNORE_LABEL


def prepare_synthia(image_dir: Path, label_dir: Path, output_path: Path):
    """
    Prepare SYNTHIA dataset in detectron2 data format by creating `sem_seg_file` for each image in output_path
    """
    # https://github.com/microsoft/ProDA/blob/main/data/synthia_dataset.py#L39
    valid_classes = [3, 4, 2, 21, 5, 7, 15, 9, 6, 16, 1, 10, 17, 8, 18, 19, 20, 12, 11]
    synthia_2_cityscape_mapping = dict(zip(valid_classes, range(19)))

    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_dir.glob("*.png")):
        image_name = image_path.name
        image_label = cv2.imread(str(label_dir / image_name), cv2.IMREAD_UNCHANGED)
        image_label = image_label[:, :, 2]
        encoded_image_label = encode_segmap(image_label, synthia_2_cityscape_mapping)
        cv2.imwrite(str(output_path / image_name), encoded_image_label)

    logging.info('Prepared semantic segmentation files for SYNTHIA dataset.')


def encode_segmap(label: np.array, label_mapping: dict):
    """
    Encodes integer labels in segmentation map according to label_mapping
    # https://github.com/microsoft/ProDA/blob/main/data/synthia_dataset.py#L111
    """
    label_copy = IGNORE_LABEL * np.ones(label.shape, dtype=np.uint8)
    for k, v in label_mapping.items():
        label_copy[label == k] = v
    return label_copy


def prepare_gtav(image_dir: Path, label_dir: Path, output_path: Path):
    pass


@click.command()
@click.option('--dataset', '-d', required=True, type=str)
@click.option('--image-dir', '-i')
@click.option('--label-dir', '-l')
@click.option('--output-path', '-o')
def main(dataset, image_dir, label_dir, output_path):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_path = Path(output_path)

    logging.info("Preparing %s dataset..." % dataset)

    if dataset == 'synthia':
        prepare_synthia(image_dir, label_dir, output_path)
    elif dataset == 'gta5':
        prepare_gtav(image_dir, label_dir, output_path)
    else:
        logging.error('Dataset %s does have a preparation function' % dataset)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
