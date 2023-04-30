import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
import evaluation.Iou


def read_nii(path):
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img), spacing


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def process_label(label):
    rv = label == 1
    myo = label == 2
    lv = label == 3

    return rv, myo, lv


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        print(hd95)
        return hd95
    else:
        return 0


def precision(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        precision = binary.precision(pred, gt)
        print(precision)
        return precision
    else:
        return 0


def recall(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        recall = binary.recall(pred, gt)
        print(recall)
        return recall
    else:
        return 0


def iou(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        iou = evaluation.Iou.iou(pred, gt)
        print(iou)
        return iou
    else:
        return 0


def inference():
    label_list = sorted(glob.glob(
        os.path.join(
            "/media/lab429/data1/CYCY/TCCoNet/DATASET/TCCoNet_raw/TCCoNet_raw_data/Task12_ACDC/labelsTs",
            '*nii.gz')))
    dest_file = r"/media/lab429/data1/CYCY/Result/new/ACDC/TransFuse/fold_2/model_final_checkpoint"
    infer_list = sorted(
        glob.glob((os.path.join(dest_file, '*nii.gz'))))

    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_rv = []
    Dice_myo = []
    Dice_lv = []

    hd_rv = []
    hd_myo = []
    hd_lv = []

    pre_rv = []
    pre_myo = []
    pre_lv = []

    rec_rv = []
    rec_myo = []
    rec_lv = []

    iou_rv = []
    iou_myo = []
    iou_lv = []

    file = dest_file
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file + '/dice_pre.txt', 'w')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path)
        print(infer_path)
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, spacing = read_nii(label_path)
        infer, spacing = read_nii(infer_path)
        label_rv, label_myo, label_lv = process_label(label)
        infer_rv, infer_myo, infer_lv = process_label(infer)

        Dice_rv.append(dice(infer_rv, label_rv))
        Dice_myo.append(dice(infer_myo, label_myo))
        Dice_lv.append(dice(infer_lv, label_lv))

        hd_rv.append(hd(infer_rv, label_rv))
        hd_myo.append(hd(infer_myo, label_myo))
        hd_lv.append(hd(infer_lv, label_lv))

        pre_rv.append(precision(infer_rv, label_rv))
        pre_myo.append(precision(infer_rv, label_rv))
        pre_lv.append(precision(infer_rv, label_rv))

        rec_rv.append(recall(infer_rv, label_rv))
        rec_myo.append(recall(infer_rv, label_rv))
        rec_lv.append(recall(infer_rv, label_rv))

        iou_rv.append(iou(infer_rv, label_rv))
        iou_myo.append(iou(infer_rv, label_rv))
        iou_lv.append(iou(infer_rv, label_rv))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_rv[-1]))
        fw.write('Dice_myo: {:.4f}\n'.format(Dice_myo[-1]))
        fw.write('Dice_lv: {:.4f}\n'.format(Dice_lv[-1]))

        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))

        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('pre_rv: {:.4f}\n'.format(pre_rv[-1]))
        fw.write('pre_myo: {:.4f}\n'.format(pre_myo[-1]))
        fw.write('pre_lv: {:.4f}\n'.format(pre_lv[-1]))
        fw.write(infer_path.split('/')[-1] + '\n')

        fw.write('rec_rv: {:.4f}\n'.format(rec_rv[-1]))
        fw.write('rec_myo: {:.4f}\n'.format(rec_myo[-1]))
        fw.write('rec_lv: {:.4f}\n'.format(rec_lv[-1]))
        fw.write(infer_path.split('/')[-1] + '\n')

        fw.write('iou_rv: {:.4f}\n'.format(iou_rv[-1]))
        fw.write('iou_myo: {:.4f}\n'.format(iou_myo[-1]))
        fw.write('iou_lv: {:.4f}\n'.format(iou_lv[-1]))
        fw.write('*' * 20 + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv' + str(np.mean(Dice_rv)) + '\n')
    fw.write('Dice_myo' + str(np.mean(Dice_myo)) + '\n')
    fw.write('Dice_lv' + str(np.mean(Dice_lv)) + '\n')
    fw.write('Mean_HD\n')
    fw.write('HD_rv' + str(np.mean(hd_rv)) + '\n')
    fw.write('HD_myo' + str(np.mean(hd_myo)) + '\n')
    fw.write('HD_lv' + str(np.mean(hd_lv)) + '\n')
    fw.write('Mean_Precision\n')
    fw.write('Precision_rv' + str(np.mean(pre_rv)) + '\n')
    fw.write('Precision_myo' + str(np.mean(pre_myo)) + '\n')
    fw.write('Precision_lv' + str(np.mean(pre_lv)) + '\n')
    fw.write('Mean_Recall\n')
    fw.write('Recall_rv' + str(np.mean(rec_rv)) + '\n')
    fw.write('Recall_myo' + str(np.mean(rec_myo)) + '\n')
    fw.write('Recall_lv' + str(np.mean(rec_lv)) + '\n')
    fw.write('Mean_mIou\n')
    fw.write('mIou_rv' + str(np.mean(iou_rv)) + '\n')
    fw.write('mIou_myo' + str(np.mean(iou_myo)) + '\n')
    fw.write('mIou_lv' + str(np.mean(iou_lv)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))

    avg_hd = []
    avg_hd.append(np.mean(hd_rv))
    avg_hd.append(np.mean(hd_myo))
    avg_hd.append(np.mean(hd_lv))

    avg_pre = []
    avg_pre.append(np.mean(pre_rv))
    avg_pre.append(np.mean(pre_myo))
    avg_pre.append(np.mean(pre_lv))

    avg_rec = []
    avg_rec.append(np.mean(rec_rv))
    avg_rec.append(np.mean(rec_myo))
    avg_rec.append(np.mean(rec_lv))

    avg_iou = []
    avg_iou.append(np.mean(iou_rv))
    avg_iou.append(np.mean(iou_myo))
    avg_iou.append(np.mean(iou_lv))

    fw.write('DSC:' + str(np.mean(dsc)) + '\n')
    fw.write('HD:' + str(np.mean(avg_hd)) + '\n')
    fw.write('mPrecision:' + str(np.mean(avg_pre)) + '\n')
    fw.write('mRecall:' + str(np.mean(avg_rec)) + '\n')
    fw.write('mIou:' + str(np.mean(avg_iou)) + '\n')

    print('done')


if __name__ == '__main__':
    inference()
