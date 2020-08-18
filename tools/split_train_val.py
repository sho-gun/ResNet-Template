import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dataset_dir', type=str, required=True)
parser.add_argument('--ratio', type=float, required=False, default=0.1, help='Value of "# of val set" / "# of train set"')

def _main(args):
    dataset_dir = os.path.expanduser(args.input_dataset_dir)
    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels')

    ratio = args.ratio

    train_list = open(os.path.join(dataset_dir, 'train.lst'), 'w')
    val_list = open(os.path.join(dataset_dir, 'val.lst'), 'w')

    image_list = sorted(os.listdir(img_dir))
    label_list = sorted(os.listdir(lbl_dir))

    val_index = round(1 / ratio)
    for i, (image, label) in enumerate(zip(image_list, label_list)):
        if i % val_index == 0:
            val_list.write('{} {}\n'.format(os.path.join(img_dir, image), os.path.join(lbl_dir, label)))
        else:
            train_list.write('{} {}\n'.format(os.path.join(img_dir, image), os.path.join(lbl_dir, label)))

    train_list.close()
    val_list.close()

if __name__ == '__main__':
    _main(parser.parse_args())
