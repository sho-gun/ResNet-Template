import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dataset_dir', type=str, required=True)

def _main(args):
    dataset_dir = os.path.expanduser(args.input_dataset_dir)
    lbl_dir = os.path.join(dataset_dir, 'labels')
    classes_file = os.path.join(dataset_dir, 'classes.txt')

    with open(classes_file, 'r') as class_names:
        classes = {k: {'name': v.strip(), 'num': 0} for k, v in enumerate(class_names.readlines())}

    label_list = sorted(os.listdir(lbl_dir))
    total = 0

    for label_file in label_list:
        with open(os.path.join(lbl_dir, label_file), 'r') as label:
            label_id = int(label.readline().strip())
            classes[label_id]['num'] += 1
            total += 1

    for class_info in classes.values():
        print(class_info['name'])
        print('\tnumber of data: {}'.format(class_info['num']))
        print('\t{}% of the dataset'.format(round(100 * class_info['num']/total, 2)))

if __name__ == '__main__':
    _main(parser.parse_args())
