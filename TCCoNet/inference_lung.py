import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
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
    cancer = label == 1
    return cancer


def inference():
    label_list = sorted(glob.glob(os.path.join(
        r"G:/CY/CodesOfCy/new/TCCoNet-main/DATASET/TCCoNet_raw/TCCoNet_raw_data/Task06_Lung/labelsTs",
        '*nii.gz')))
    dest_file = r"G:\CY\results\Second\TCCoNet\lung\UNETR\model_final_checkpoint"
    infer_list = sorted(
        glob.glob(os.path.join(dest_file, '*nii.gz')))
    print("loading success...")
    Dice_cancer = []
    HD_cancer = []
    file = dest_file
    fw = open(file + '/dice_pre.txt', 'w')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_cancer = process_label(label)
        infer_cancer = process_label(infer)
        Dice_cancer.append(dice(infer_cancer, label_cancer))
        HD_cancer.append(hd(infer_cancer, label_cancer))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('HD_cancer: {:.4f}\n'.format(HD_cancer[-1]))
        fw.write('Dice_cancer: {:.4f}\n'.format(Dice_cancer[-1]))
    dsc = []
    avg_hd = []
    dsc.append(np.mean(Dice_cancer))

    avg_hd.append(np.mean(HD_cancer))

    fw.write('Dice_cancer' + str(np.mean(Dice_cancer)) + ' ' + '\n')

    fw.write('HD_cancer' + str(np.mean(HD_cancer)) + ' ' + '\n')

    fw.write('Dice' + str(np.mean(dsc)) + ' ' + '\n')
    fw.write('HD' + str(np.mean(avg_hd)) + ' ' + '\n')


if __name__ == '__main__':
    inference()
