import os
import shutil
import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images_dir', type=str, required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('-c', '--class_list', type=str, required=True)

def _main(args):
    images_dir = os.path.expanduser(args.images_dir)
    output_dir = os.path.expanduser(args.output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    output_lbl_dir = os.path.join(output_dir, 'labels')

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    class_dict = {k: v.strip() for k, v in enumerate(open(args.class_list))}

    for image_file in sorted(os.listdir(images_dir)):
        basename, ext = os.path.splitext(image_file)
        output_lbl_file = os.path.join(output_lbl_dir, basename + '.txt')

        if os.path.exists(output_lbl_file):
            continue

        if ext in ['.jpg', '.png']:
            thread = threading.Thread(target=create_label_file, args=(output_lbl_file, class_dict))
            thread.start()

            image = Image.open(os.path.join(images_dir, image_file))
            image = np.array(image)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.tight_layout()
            plt.show()

            if not image_file in os.listdir(output_img_dir):
                shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_img_dir, image_file))

    shutil.copy(args.class_list, os.path.join(output_dir, os.path.basename(args.class_list)))

def create_label_file(label_path, class_dict):
    for k, v in class_dict.items():
        print(k, v)
    class_id = int(input('>>'))

    if not class_id in class_dict.keys():
        print('ERROR: Input number should be in class id')
        exit(1)

    with open(label_path, 'w') as label_file:
        label_file.write('{}'.format(class_id))

    plt.close()
    print('')

if __name__ == '__main__':
    _main(parser.parse_args())
