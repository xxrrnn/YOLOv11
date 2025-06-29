import os
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np

def convert_coco_json(json_dir, save_dir, class_label_file):
    # Step 1: Read the official class list to create the name -> 0-79 index mapping.
    with open(class_label_file) as f:
        class_names = [line.strip() for line in f.readlines()]
    name_to_id = {name: i for i, name in enumerate(class_names)}

    jsons = [f for f in os.listdir(json_dir) if f.endswith('.json') and 'instances_val2017' in f]
    for json_file in jsons:
        if '2017' in json_file:
            split_name = json_file.replace('instances_', '').replace('.json', '')
            fn = Path(save_dir) / 'labels' / split_name
            fn.mkdir(parents=True, exist_ok=True)
            with open(os.path.join(json_dir, json_file)) as f:
                data = json.load(f)

            # Step 2: Create a mapping from the JSON's category ID to the category NAME.
            json_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

            images = {'%g' % x['id']: x for x in data['images']}
            for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
                if x['iscrowd']:
                    continue

                img = images['%g' % x['image_id']]
                h, w, f = img['height'], img['width'], img['file_name']

                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize w
                box[[1, 3]] /= h  # normalize h

                # Step 3: Look up the final 0-79 ID using the name.
                category_name = json_id_to_name.get(x['category_id'])
                if category_name and category_name in name_to_id:
                    cls = name_to_id[category_name]  # Use the correct mapping from the label file
                    if (box[2] > 0) and (box[3] > 0):
                        line = cls, *box
                        with open(fn / (f.split('.')[0] + '.txt'), 'a') as file:
                            file.write(('%g ' * len(line)).rstrip() % line + '\n')

if __name__ == '__main__':
    # Define paths
    coco_root = Path('Dataset/coco2017')
    annotations_dir = coco_root / 'annotations'
    class_label_file = coco_root / 'labels' / 'coco-labels.txt'

    # Convert annotations
    convert_coco_json(annotations_dir, coco_root, class_label_file)

    # Create image list files
    for split in ['val2017']:
        with open(coco_root / f'{split}.txt', 'w') as f:
            for img_path in sorted((coco_root / 'images' / split).glob('*.jpg')):
                f.write(f'./{img_path.relative_to(Path("Dataset/coco2017"))}\n')

    print("Conversion and file list generation complete.")