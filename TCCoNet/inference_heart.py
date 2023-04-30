import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def new_dice(pred, label):
    tp_hard = np.sum((pred == 1).astype(np.float) * (label == 1).astype(np.float))
    fp_hard = np.sum((pred == 1).astype(np.float) * (label != 1).astype(np.float))
    fn_hard = np.sum((pred != 1).astype(np.float) * (label == 1).astype(np.float))
    return 2 * tp_hard / (2 * tp_hard + fp_hard + fn_hard)


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        return hd95
    else:
        return 0


def process_label(label):
    heart = label == 1
    return heart


def inference():
    label_list = sorted(glob.glob(os.path.join(
        r"G:/CY/CodesOfCy/new/TCCoNet-main/DATASET/TCCoNet_raw/TCCoNet_raw_data/Task05_Heart/labelsTs",
        '*nii.gz')))
    dest_file = r"G:\CY\results\Third\Heart\UNETR\model_best"
    infer_list = sorted(
        glob.glob(os.path.join(dest_file, '*nii.gz')))
    print("loading success...")
    Dice_heart = []
    HD_heart = []
    file = dest_file
    fw = open(file + '/dice_pre.txt', 'w')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_heart = process_label(label)
        infer_heart = process_label(infer)
        Dice_heart.append(dice(infer_heart, label_heart))
        HD_heart.append(hd(infer_heart, label_heart))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('HD_heart: {:.4f}\n'.format(HD_heart[-1]))
        fw.write('Dice_heart: {:.4f}\n'.format(Dice_heart[-1]))
    dsc = []
    avg_hd = []
    dsc.append(np.mean(Dice_heart))

    avg_hd.append(np.mean(HD_heart))

    fw.write('Dice_heart' + str(np.mean(Dice_heart)) + ' ' + '\n')
    fw.write('HD_heart' + str(np.mean(HD_heart)) + ' ' + '\n')

    fw.write('Dice' + str(np.mean(dsc)) + ' ' + '\n')
    fw.write('HD' + str(np.mean(avg_hd)) + ' ' + '\n')


if __name__ == '__main__':
    inference()
