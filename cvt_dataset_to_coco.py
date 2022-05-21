"""
Script to convert packaging dataset into COCO-like format.
"""
import argparse
import json
import pathlib

import cv2
import fiftyone as fo

LABEL_FIELD = "ground_truth"
# Since this dataset only has one class, it's easier to just hardcode the label name
LABEL_NAME = "packaging"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=str, help="Path to the dataset root directory"
    )
    parser.add_argument("--output-dir", type=str, help="Path to the output directory")
    return parser.parse_args()


def get_all_images_pathnames(data_root: pathlib.Path):
    pathnames = []
    for format in ["jpg", "png", "jpeg"]:
        pathnames += list(data_root.glob(f"**/*.{format}"))
    return pathnames


def main(args):
    data_root = pathlib.Path(args.data_root)
    # This assumes that the dataset is organized as follows:
    images_path = data_root / "images"
    annotations_path = data_root / "annotations"
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset = fo.Dataset(name="packaging_dataset")
    all_images_pathnames = get_all_images_pathnames(images_path)
    print(f"Found {len(all_images_pathnames)} images")
    for image_path in all_images_pathnames:
        labels_path = annotations_path / image_path.name.replace(".jpg", ".json")
        with open(labels_path, "r") as f:
            labels = json.load(f)
        image = cv2.imread(str(image_path))
        img_width, img_height = image.shape[:2]
        sample = fo.Sample(filepath=image_path)
        detections = []
        for bbox in labels["bboxes"]:
            # This dataset only has one class
            xmin, ymin, xmax, ymax = bbox
            normalized_top_lef_x = xmin / img_width
            normalized_top_left_y = ymin / img_height
            normalized_width = (xmax - xmin) / img_width
            normalized_height = (ymax - ymin) / img_height
            new_bounding_box = [
                normalized_top_lef_x,
                normalized_top_left_y,
                normalized_width,
                normalized_height,
            ]
            detection = fo.Detection(bounding_box=new_bounding_box, label=LABEL_NAME)
            detections.append(detection)
        sample[LABEL_FIELD] = fo.Detections(detections=detections)
        sample["metadata"] = fo.Metadata(width=img_width, height=img_height)
        dataset.add_sample(sample)
    dataset.export(
        export_dir=str(output_dir),
        dataset_type=fo.types.COCODetectionDataset,
        label_field=LABEL_FIELD,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
