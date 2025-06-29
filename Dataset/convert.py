import os
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np

def convert_coco_json(json_dir, save_dir, class_label_file):
    """
    Converts COCO JSON format annotations to YOLO .txt format.

    Args:
        json_dir (str or Path): Path to the directory containing COCO JSON files.
        save_dir (str or Path): Path to the directory where the YOLO labels will be saved.
        class_label_file (str or Path): Path to the file containing class names, one per line.
    """
    # Step 1: Read the official class list to create a mapping from class name to a 0-79 index.
    # This ensures the class IDs in the output files are sequential and zero-based.
    with open(class_label_file) as f:
        class_names = [line.strip() for line in f.readlines()]
    name_to_id = {name: i for i, name in enumerate(class_names)}

    # Filter for the specific COCO validation instance annotation file.
    jsons = [f for f in os.listdir(json_dir) if f.endswith('.json') and 'instances_val2017' in f]
    for json_file in jsons:
        # Create a directory for the output labels, named after the data split (e.g., 'val2017').
        if '2017' in json_file:
            split_name = json_file.replace('instances_', '').replace('.json', '')
            fn = Path(save_dir) / 'labels' / split_name
            fn.mkdir(parents=True, exist_ok=True)
            with open(os.path.join(json_dir, json_file)) as f:
                data = json.load(f)

            # Step 2: Create a mapping from the JSON's internal category ID to the category NAME.
            # COCO JSON category IDs are not necessarily sequential or zero-based, so this mapping is needed.
            json_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

            # Create a dictionary of image info for quick lookup by image ID.
            images = {'%g' % x['id']: x for x in data['images']}

            # Process each annotation in the JSON file.
            for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
                if x['iscrowd']:
                    continue

                # Get image details (height, width, filename) using the image_id from the annotation.
                img = images['%g' % x['image_id']]
                h, w, f = img['height'], img['width'], img['file_name']

                # Convert bounding box from [x, y, w, h] (top-left corner) to [x_center, y_center, w, h] (YOLO format).
                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                # Normalize coordinates by image dimensions.
                box[[0, 2]] /= w  # normalize w
                box[[1, 3]] /= h  # normalize h

                # Step 3: Look up the final 0-79 class ID using the category name.
                category_name = json_id_to_name.get(x['category_id'])
                # Ensure the category from the JSON exists in our official class list.
                if category_name and category_name in name_to_id:
                    cls = name_to_id[category_name]  # Use the 0-79 mapping from the label file.
                    # Ensure the bounding box has a valid area (width and height are positive).
                    if (box[2] > 0) and (box[3] > 0):
                        # Format the line for the label file: class_id, x_center, y_center, width, height.
                        line = cls, *box
                        # Write the label to a .txt file with the same name as the image.
                        with open(fn / (f.split('.')[0] + '.txt'), 'a') as file:
                            file.write(('%g ' * len(line)).rstrip() % line + '\n')

if __name__ == '__main__':
    # Define paths for the COCO dataset components.
    coco_root = Path('./coco2017')
    annotations_dir = coco_root / 'annotations'
    class_label_file = coco_root / 'coco_labels.txt'

    # Run the conversion from COCO JSON to YOLO .txt format.
    convert_coco_json(annotations_dir, coco_root, class_label_file)

    # Create .txt files that list the image paths for each data split (e.g., val2017.txt).
    # These files are used by data loaders during training to find the images.
    for split in ['val2017']:
        with open(coco_root / f'{split}.txt', 'w') as f:
            # Find all .jpg images in the split's image directory.
            for img_path in sorted((coco_root / 'images' / split).glob('*.jpg')):
                # Write the relative path of each image to the list file.
                f.write(f'./{img_path.relative_to(Path("./coco2017"))}\n')

    print("Conversion and file list generation complete.")