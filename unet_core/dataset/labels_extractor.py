import cv2
import numpy as np
from pathlib import Path


def prepare_unet_data(data_path):
    data_path = Path(data_path)
    images_dir = data_path / "images" / "train"
    labels_dir = data_path / "labels" / "train"

    output_images_dir = data_path / "shaped" / "images"
    output_labels_dir = data_path / "shaped" / "masks"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images_dir.glob("*.tif"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Nie udało się wczytać obrazu: {image_path}")
            continue

        height, width = image.shape[:2]
        base_name = image_path.stem
        label_path = labels_dir / f"{base_name}.txt"

        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for idx, line in enumerate(lines):
            try:
                points = [tuple(map(float, p.split(","))) for p in line.split(";")]
                x_coords, y_coords = zip(*points)

                xmin = max(0, int(min(x_coords)))
                xmax = min(width, int(max(x_coords)))
                ymin = max(0, int(min(y_coords)))
                ymax = min(height, int(max(y_coords)))

                if xmax <= xmin or ymax <= ymin:
                    print(f"⚠️ Pominięto pusty wycinek w pliku {label_path}, linia {idx}")
                    continue

                cropped_img = image[ymin:ymax, xmin:xmax]
                output_img_path = output_images_dir / f"{base_name}_{idx}.jpg"
                cv2.imwrite(str(output_img_path), cropped_img)

                mask = np.zeros((height, width), dtype=np.uint8)
                points_np = np.array([points], dtype=np.int32)
                cv2.fillPoly(mask, points_np, color=255)
                cropped_mask = mask[ymin:ymax, xmin:xmax]
                output_mask_path = output_labels_dir / f"{base_name}_{idx}.png"
                cv2.imwrite(str(output_mask_path), cropped_mask)

            except Exception as e:
                print(f"❌ Błąd w pliku {label_path}, linia {idx}: {e}")
