import os
import logging
import argparse
import numpy as np
from skimage import io
from skimage.util import img_as_float
from tqdm import tqdm
import sys
import time
import shutil
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Argument parser
parser = argparse.ArgumentParser("Sam infer")
parser.add_argument("--root_folder", type=str, default='../sam_output/DUTS/vit_h/', help="Location of the models to load in.")
parser.add_argument('--ground_truth_folder', type=str, default='../dataset/DUTS/test_masks', help='Ground truth folder')
parser.add_argument('--job_name', type=str, default='DUTS_vit_h', help='Job name')

args, unparsed = parser.parse_known_args()

# Create experiment directory
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

args.job_name = './results/' + time.strftime("%Y%m%d-%H%M%S-") + str(args.job_name)
create_exp_dir(args.job_name)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Metrics calculations
def calculate_f_measure(prediction, ground_truth, beta=1):
    prediction = prediction.astype(np.bool)
    ground_truth = ground_truth.astype(np.bool)

    tp = np.sum(prediction & ground_truth)
    fp = np.sum(prediction & ~ground_truth)
    fn = np.sum(~prediction & ground_truth)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_measure = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)

    return f_measure

def dice_coefficient(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice

def cal_iou(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt + pred)
    iou = intersection / (union - intersection + 1e-8)
    return iou

def mean_absolute_error(gt, pred):
    mae = np.mean(np.abs(pred - gt))
    return mae

def binary_loader(path):
    assert os.path.exists(path), f"`{path}` does not exist."
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")

def normalize_pil(pre, gt):
    gt = np.asarray(gt)
    pre = np.asarray(pre)
    gt = gt / (gt.max() + 1e-8)
    gt = np.where(gt > 0.5, 1, 0)
    max_pre = pre.max()
    min_pre = pre.min()
    if max_pre == min_pre:
        pre = pre / 255
    else:
        pre = (pre - min_pre) / (max_pre - min_pre)
    return pre, gt

def find_best_metrics(prediction_folder, ground_truth_file):
    ground_truth = binary_loader(ground_truth_file)
    best_dice = 0
    best_f_measure = 0
    best_mae = 1
    best_iou = 0
    best_prediction_file = None

    for prediction_file in os.listdir(prediction_folder):
        if prediction_file == 'metadata.csv':
            continue
        prediction_path = os.path.join(prediction_folder, prediction_file)
        prediction = binary_loader(prediction_path)
        if prediction.size != ground_truth.size:
            prediction = prediction.resize(ground_truth.size, Image.BILINEAR)
        prediction1, ground_truth1 = normalize_pil(pre=prediction, gt=ground_truth)

        try:
            f_dice = dice_coefficient(ground_truth1, prediction1)
            f_measure = calculate_f_measure(ground_truth1, prediction1)
            mae_measure = mean_absolute_error(ground_truth1, prediction1)
            iou_measure = cal_iou(ground_truth1, prediction1)
        except ValueError:
            print(prediction_file)
            print(prediction_folder)

        if f_dice > best_dice:
            best_dice = f_dice
            best_prediction_file = prediction_file
            best_mae = mae_measure
            best_f_measure = f_measure
            best_iou = iou_measure

    return best_dice, best_f_measure, best_prediction_file, best_mae, best_iou

def main():
    total_dice = 0
    total_f_measure = 0
    total_mae = 0
    total_iou = 0

    folder_count = 0

    folders = os.listdir(args.root_folder)
    for folder in tqdm(folders, desc="Processing folders"):
        prediction_folder = os.path.join(args.root_folder, folder)
        ground_truth_file = os.path.join(args.ground_truth_folder, folder + ".png")

        best_dice, best_f_measure, best_prediction_file, best_mae, best_iou = find_best_metrics(prediction_folder, ground_truth_file)
        print(f"prediction_folder: {prediction_folder}")
        print(f"best_prediction_file: {best_prediction_file}")
        print(f"best_dice: {best_dice}")
        print(f"best_f1_measure: {best_f_measure}")
        print(f"best_mae: {best_mae}")
        print(f"best_iou: {best_iou}")

        total_dice += best_dice
        total_f_measure += best_f_measure
        total_mae += best_mae
        total_iou += best_iou
        folder_count += 1

    average_dice = total_dice / folder_count
    average_f1_measure = total_f_measure / folder_count
    average_mae_measure = total_mae / folder_count
    average_iou_measure = total_iou / folder_count

    logging.info(f"Average best Dice: {average_dice:.4f}, Average best F1: {average_f1_measure:.4f}, Average best MAE: {average_mae_measure:.4f}, Average best mIoU: {average_iou_measure:.4f}, Number: {folder_count}")

if __name__ == "__main__":
    main()
