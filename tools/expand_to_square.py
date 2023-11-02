import argparse
import json
import os
import re

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True, type=str, help='The input SVIT JSON file')
parser.add_argument('--image_dir', required=True, type=str, help='The directory of images')
parser.add_argument('--output_file', required=True, type=str, help='The output SVIT JSON file')

def expand_to_square(box, w, h):
    if w == h:
        return box
    
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box

def normalize_bbox(box, w, h):
    if w > h:
        return [round(i / w, 3) for i in box]
    
    return [round(i / h, 3) for i in box]

if __name__ == "__main__":
    args = parser.parse_args()

    bbox_regex = r"\[[0-9.]+, [0-9.]+, [0-9.]+, [0-9.]+\]"
    images = dict()

    with open(args.input_file, 'r') as fin:
        data = json.load(fin)

    pbar = tqdm(data)
    for item in pbar:
        image_file_name = f"{item['image_id']}.jpg"

        if image_file_name in images:
            width, height = images[image_file_name]
        else:
            image = Image.open(os.path.join(args.image_dir, image_file_name))
            width, height = image.size
            images[image_file_name] = width, height

        for conversation in item['conversations']:
            for qa in conversation['content']:
                regex = re.compile(r'\[[0-9.]+, [0-9.]+, [0-9.]+, [0-9.]+\]')
                matches = re.findall(regex, qa['value'])
                new_value = qa['value']
                
                for match in matches:
                    try:
                        x1, y1, x2, y2 = [float(part.strip()) for part in match.strip('[]').split(',')]
                        original_bbox = round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)
                        expanded_bbox = expand_to_square(original_bbox, width, height)
                        new_x1, new_y1, new_x2, new_y2 = normalize_bbox(expanded_bbox, width, height)
                        new_bbox_string = f'[{new_x1}, {new_y1}, {new_x2}, {new_y2}]'
                        new_value = new_value.replace(match, new_bbox_string)
                    except:
                        pass

                qa['value'] = new_value

    with open(args.output_file, "w+") as fout:
        fout.write(json.dumps(data, indent=4))