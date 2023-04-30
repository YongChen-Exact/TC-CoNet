#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from TCCoNet.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "TCCoNetPlans"

    overwrite_plans = {
        'TCCoNetTrainerV2_2': ["TCCoNetPlans", "TCCoNetPlansisoPatchesInVoxels"], # r
        'TCCoNetTrainerV2': ["TCCoNetPlansnonCT", "TCCoNetPlansCT2", "TCCoNetPlansallConv3x3",
                            "TCCoNetPlansfixedisoPatchesInVoxels", "TCCoNetPlanstargetSpacingForAnisoAxis",
                            "TCCoNetPlanspoolBasedOnSpacing", "TCCoNetPlansfixedisoPatchesInmm", "TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_warmup': ["TCCoNetPlans", "TCCoNetPlansv2.1", "TCCoNetPlansv2.1_big", "TCCoNetPlansv2.1_verybig"],
        'TCCoNetTrainerV2_cycleAtEnd': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_cycleAtEnd2': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_reduceMomentumDuringTraining': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_graduallyTransitionFromCEToDice': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_independentScalePerAxis': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_Mish': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_Ranger_lr3en4': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_fp32': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_GN': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_momentum098': ["TCCoNetPlans", "TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_momentum09': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_DP': ["TCCoNetPlansv2.1_verybig"],
        'TCCoNetTrainerV2_DDP': ["TCCoNetPlansv2.1_verybig"],
        'TCCoNetTrainerV2_FRN': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_resample33': ["TCCoNetPlansv2.3"],
        'TCCoNetTrainerV2_O2': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_ResencUNet': ["TCCoNetPlans_FabiansResUNet_v2.1"],
        'TCCoNetTrainerV2_DA2': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_allConv3x3': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_ForceBD': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_ForceSD': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_LReLU_slope_2en1': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_lReLU_convReLUIN': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_ReLU': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_ReLU_biasInSegOutput': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_ReLU_convReLUIN': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_lReLU_biasInSegOutput': ["TCCoNetPlansv2.1"],
        #'TCCoNetTrainerV2_Loss_MCC': ["TCCoNetPlansv2.1"],
        #'TCCoNetTrainerV2_Loss_MCCnoBG': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_Loss_DicewithBG': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_Loss_Dice_LR1en3': ["TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_Loss_Dice': ["TCCoNetPlans", "TCCoNetPlansv2.1"],
        'TCCoNetTrainerV2_Loss_DicewithBG_LR1en3': ["TCCoNetPlansv2.1"],
        # 'TCCoNetTrainerV2_fp32': ["TCCoNetPlansv2.1"],
        # 'TCCoNetTrainerV2_fp32': ["TCCoNetPlansv2.1"],
        # 'TCCoNetTrainerV2_fp32': ["TCCoNetPlansv2.1"],
        # 'TCCoNetTrainerV2_fp32': ["TCCoNetPlansv2.1"],
        # 'TCCoNetTrainerV2_fp32': ["TCCoNetPlansv2.1"],

    }

    trainers = ['TCCoNetTrainer'] + ['TCCoNetTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'TCCoNetTrainerNewCandidate24_2',
        'TCCoNetTrainerNewCandidate24_3',
        'TCCoNetTrainerNewCandidate26_2',
        'TCCoNetTrainerNewCandidate27_2',
        'TCCoNetTrainerNewCandidate23_always3DDA',
        'TCCoNetTrainerNewCandidate23_corrInit',
        'TCCoNetTrainerNewCandidate23_noOversampling',
        'TCCoNetTrainerNewCandidate23_softDS',
        'TCCoNetTrainerNewCandidate23_softDS2',
        'TCCoNetTrainerNewCandidate23_softDS3',
        'TCCoNetTrainerNewCandidate23_softDS4',
        'TCCoNetTrainerNewCandidate23_2_fp16',
        'TCCoNetTrainerNewCandidate23_2',
        'TCCoNetTrainerVer2',
        'TCCoNetTrainerV2_2',
        'TCCoNetTrainerV2_3',
        'TCCoNetTrainerV2_3_CE_GDL',
        'TCCoNetTrainerV2_3_dcTopk10',
        'TCCoNetTrainerV2_3_dcTopk20',
        'TCCoNetTrainerV2_3_fp16',
        'TCCoNetTrainerV2_3_softDS4',
        'TCCoNetTrainerV2_3_softDS4_clean',
        'TCCoNetTrainerV2_3_softDS4_clean_improvedDA',
        'TCCoNetTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'TCCoNetTrainerV2_3_softDS4_radam',
        'TCCoNetTrainerV2_3_softDS4_radam_lowerLR',

        'TCCoNetTrainerV2_2_schedule',
        'TCCoNetTrainerV2_2_schedule2',
        'TCCoNetTrainerV2_2_clean',
        'TCCoNetTrainerV2_2_clean_improvedDA_newElDef',

        'TCCoNetTrainerV2_2_fixes', # running
        'TCCoNetTrainerV2_BN', # running
        'TCCoNetTrainerV2_noDeepSupervision', # running
        'TCCoNetTrainerV2_softDeepSupervision', # running
        'TCCoNetTrainerV2_noDataAugmentation', # running
        'TCCoNetTrainerV2_Loss_CE', # running
        'TCCoNetTrainerV2_Loss_CEGDL',
        'TCCoNetTrainerV2_Loss_Dice',
        'TCCoNetTrainerV2_Loss_DiceTopK10',
        'TCCoNetTrainerV2_Loss_TopK10',
        'TCCoNetTrainerV2_Adam', # running
        'TCCoNetTrainerV2_Adam_TCCoNetTrainerlr', # running
        'TCCoNetTrainerV2_SGD_ReduceOnPlateau', # running
        'TCCoNetTrainerV2_SGD_lr1en1', # running
        'TCCoNetTrainerV2_SGD_lr1en3', # running
        'TCCoNetTrainerV2_fixedNonlin', # running
        'TCCoNetTrainerV2_GeLU', # running
        'TCCoNetTrainerV2_3ConvPerStage',
        'TCCoNetTrainerV2_NoNormalization',
        'TCCoNetTrainerV2_Adam_ReduceOnPlateau',
        'TCCoNetTrainerV2_fp16',
        'TCCoNetTrainerV2', # see overwrite_plans
        'TCCoNetTrainerV2_noMirroring',
        'TCCoNetTrainerV2_momentum09',
        'TCCoNetTrainerV2_momentum095',
        'TCCoNetTrainerV2_momentum098',
        'TCCoNetTrainerV2_warmup',
        'TCCoNetTrainerV2_Loss_Dice_LR1en3',
        'TCCoNetTrainerV2_NoNormalization_lr1en3',
        'TCCoNetTrainerV2_Loss_Dice_squared',
        'TCCoNetTrainerV2_newElDef',
        'TCCoNetTrainerV2_fp32',
        'TCCoNetTrainerV2_cycleAtEnd',
        'TCCoNetTrainerV2_reduceMomentumDuringTraining',
        'TCCoNetTrainerV2_graduallyTransitionFromCEToDice',
        'TCCoNetTrainerV2_insaneDA',
        'TCCoNetTrainerV2_independentScalePerAxis',
        'TCCoNetTrainerV2_Mish',
        'TCCoNetTrainerV2_Ranger_lr3en4',
        'TCCoNetTrainerV2_cycleAtEnd2',
        'TCCoNetTrainerV2_GN',
        'TCCoNetTrainerV2_DP',
        'TCCoNetTrainerV2_FRN',
        'TCCoNetTrainerV2_resample33',
        'TCCoNetTrainerV2_O2',
        'TCCoNetTrainerV2_ResencUNet',
        'TCCoNetTrainerV2_DA2',
        'TCCoNetTrainerV2_allConv3x3',
        'TCCoNetTrainerV2_ForceBD',
        'TCCoNetTrainerV2_ForceSD',
        'TCCoNetTrainerV2_ReLU',
        'TCCoNetTrainerV2_LReLU_slope_2en1',
        'TCCoNetTrainerV2_lReLU_convReLUIN',
        'TCCoNetTrainerV2_ReLU_biasInSegOutput',
        'TCCoNetTrainerV2_ReLU_convReLUIN',
        'TCCoNetTrainerV2_lReLU_biasInSegOutput',
        'TCCoNetTrainerV2_Loss_DicewithBG_LR1en3',
        #'TCCoNetTrainerV2_Loss_MCCnoBG',
        'TCCoNetTrainerV2_Loss_DicewithBG',
        # 'TCCoNetTrainerV2_Loss_Dice_LR1en3',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
        # 'TCCoNetTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
