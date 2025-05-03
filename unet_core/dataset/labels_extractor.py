import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def extract_gland_polygons(xml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.iter("image"):
        image_name = image.attrib.get("name")
        base_name = os.path.splitext(image_name)[0]
        output_file_path = os.path.join(output_dir, f"{base_name}.txt")

        polygons = image.findall("polygon")
        with open(output_file_path, "w") as f:
            for polygon in polygons:
                if polygon.attrib.get("label") == "gland":
                    points = polygon.attrib.get("points")
                    f.write(points + "\n")

    print(f"Saved to file : {output_dir}")


def prepare_unet_data(data_path):
    images_dir = Path(data_path) / "images" / "train"
    labels_dir = Path(data_path) / "masks" / "train"
    polygons_dir = Path(data_path) / "gland_polygons"
    output_images_dir = Path(data_path) / "shaped" / "images"
    output_labels_dir = Path(data_path) / "shaped" / "masks"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images_dir.glob("*.tif"):
        image = np.array(Image.open(image_path))
        h, w = image.shape[:2]
        base_name = image_path.stem

        yolo_label_path = labels_dir / f"{base_name}.txt"
        polygon_path = polygons_dir / f"{base_name}.txt"
        if not yolo_label_path.exists() or not polygon_path.exists():
            continue

        with open(polygon_path, 'r') as f:
            polygon_lines = f.readlines()

        with open(yolo_label_path, 'r') as f:
            bbox_lines = f.readlines()

        if len(polygon_lines) != len(bbox_lines):
            print(f"Invalid data: {base_name} — {len(bbox_lines)} boxes, {len(polygon_lines)} mask")
            continue

        for i, (bbox_line, polygon_line) in enumerate(zip(bbox_lines, polygon_lines)):
            cls, x_center, y_center, bw, bh = map(float, bbox_line.strip().split())
            x_center *= w
            y_center *= h
            bw *= w
            bh *= h
            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)

            cropped_img = image[y1:y2, x1:x2]

            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            poly_points = np.array([list(map(float, p.split(","))) for p in polygon_line.strip().split(";")])
            poly_points[:, 0] -= x1
            poly_points[:, 1] -= y1
            cv2.fillPoly(mask, [poly_points.astype(np.int32)], color=255)

            out_img_path = output_images_dir / f"{base_name}_{i}.tif"
            out_mask_path = output_labels_dir / f"{base_name}_{i}.tif"
            Image.fromarray(cropped_img).save(out_img_path)
            Image.fromarray(mask).save(out_mask_path)

    print(f"Saved in: {output_images_dir.parent}")


xml_path = "../../data/annotations.xml"
output_dir = "../../data/gland_polygons"
data_dir = "../../data"

if __name__ == "__main__":
    extract_gland_polygons(xml_path, output_dir)
    prepare_unet_data(data_dir)
