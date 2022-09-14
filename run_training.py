""" The starting point for training our model """

import torch
import argparse
from trainer import Trainer
from inference import Inference
import time
from tqdm import tqdm
import math
import pickle
import json
import shutil
import pandas as pd
import os
import os.path
import random
from datetime import datetime
import utils
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


def run_training(N_EPOCHS, trainer, best_valid_loss, model_name, early_stopping, train_dataloader,
                 test_dataloader, curriculum_name, local_rank):
    #N_EPOCHS, trainer, best_valid_loss, model_name, early_stopping, train_example_dict,
    # test_example_dict, curriculum_name, local_rank
    t_loss, v_loss = [], []
    t_loss_local, v_loss_local = [], []
    t_loss_local_app, v_loss_local_app = [], []
    t_loss_global, v_loss_global = [], []
    t_loss_local_lbl, v_loss_local_lbl = [], []
    t_loss_global_lbl, v_loss_global_lbl = [], []
    t_loss_span, v_loss_span = [], []
    t_loss_typ, v_loss_typ = [], []
    t_loss_ctx, v_loss_ctx = [], []
    best_epoch = 0
    early_stopping_marker = []
    best_stats_list = None
    # train_dataloader = trainer.get_dataloader(train_example_dict, training=True, distributed=trainer.num_workers > 2)
    # test_dataloader = trainer.get_dataloader(test_example_dict, training=False, distributed=False)

    for epoch in range(N_EPOCHS):
        print("Epoch: {}, Training ...".format(epoch))
        start_time = time.time()

        # train_dataloader = trainer.get_single_data(train_example_dict)

        tr_l, tr_l_local, tr_l_local_app, tr_l_global, tr_l_local_lbl, tr_l_global_lbl, tr_l_span, \
            tr_l_typ, tr_l_ctx = trainer.train(train_dataloader)
        t_loss.append(tr_l)
        t_loss_local.append(tr_l_local)
        t_loss_local_app.append(tr_l_local_app)
        t_loss_global.append(tr_l_global)
        t_loss_local_lbl.append(tr_l_local_lbl)
        t_loss_global_lbl.append(tr_l_global_lbl)
        t_loss_span.append(tr_l_span)
        t_loss_typ.append(tr_l_typ)
        t_loss_ctx.append(tr_l_ctx)

        if local_rank == 0:
            vl_l, vl_l_local, vl_l_local_app, vl_l_global, vl_l_local_lbl, vl_l_global_lbl, vl_l_span, vl_l_typ, \
                vl_l_ctx, vl_stat_list = trainer.evaluate(test_dataloader)

            v_loss.append(vl_l)
            v_loss_local.append(vl_l_local)
            v_loss_local_app.append(vl_l_local_app)
            v_loss_global.append(vl_l_global)
            v_loss_local_lbl.append(vl_l_local_lbl)
            v_loss_global_lbl.append(vl_l_global_lbl)
            v_loss_span.append(vl_l_span)
            v_loss_typ.append(vl_l_typ)
            v_loss_ctx.append(vl_l_ctx)

            end_time = time.time()
            epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

            if vl_l < best_valid_loss:
                best_valid_loss = vl_l
                print("SAVING BEST MODEL!")
                torch.save(trainer.parser.state_dict(), model_name)
                best_epoch = epoch
                early_stopping_marker.append(False)
                best_stats_list = vl_stat_list
            else:
                early_stopping_marker.append(True)

            print("\n")
            print("For Curriculum", curriculum_name)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Total Loss: {tr_l:.3f}')
            print(f'\tTr Local Loss: {tr_l_local:.3f} | Tr Local Append Loss: {tr_l_local_app:.3f} | Tr Global Loss: {tr_l_global:.3f} | Tr Local Lbl Loss: {tr_l_local_lbl:.3f} | Tr Global Lbl Loss: {tr_l_global_lbl:.3f} ')
            print(f'\tTr Span Loss: {tr_l_span:.3f} | Tr Type Loss: {tr_l_typ:.3f} | Tr Ctx Lbl Loss: {tr_l_ctx:.3f}')
            print(f'\tVal. Total Loss: {vl_l:.3f}')
            print(f'\tVl Local Loss: {vl_l_local:.3f} | Vl Local Append Loss: {vl_l_local_app:.3f} | Vl Global Loss: {vl_l_global:.3f} | Vl Local Lbl Loss: {vl_l_local_lbl:.3f} | Vl Global Lbl Loss: {vl_l_global_lbl:.3f}')
            print(f'\tVl Span Loss: {vl_l_span:.3f} | Vl Type Loss: {vl_l_typ:.3f} | Vl Ctx Lbl Loss: {vl_l_ctx:.3f}')
            print("_________________________________________________________________")

            if all(early_stopping_marker[-early_stopping:]):
                print("Early stopping training as the Validation loss did NOT improve for last " + str(early_stopping) + \
                      " iterations.")
                break

    return t_loss, v_loss, t_loss_local, v_loss_local, t_loss_global, v_loss_global, t_loss_local_lbl, v_loss_local_lbl, \
           t_loss_global_lbl, v_loss_global_lbl, t_loss_span, v_loss_span, t_loss_typ, v_loss_typ, t_loss_ctx, \
           v_loss_ctx, best_valid_loss, best_epoch, t_loss_local_app, v_loss_local_app, best_stats_list


def main():
    run_time = str(int(time.time()))
    path_prefix = ""  # path of the project

    N_EPOCHS = 15
    batch_size = 8
    accumulate = 4
    lr = 1e-5
    training_file = path_prefix + "data/<training file name>"
    pre_training_file = path_prefix + "data/<file name>"
    stats_file = path_prefix + "results//<file name.csv>"
    device_num = 0
    in_dim = 768
    out_dim = 600
    skip_training = "false"
    run_inference = "false"
    sigmoid_threshold = 0.5
    early_stopping = 2
    experiment_number = "-1"
    do_self_mha = True
    predict_contextual_relationship, context_attention = "true", "true"
    increase_token_type, increase_positional_embeddings = "false", "false"
    p_local_relations, p_global_relations, p_segments, p_edu_type = True, True, True, True
    debug_mode = "false"
    train_distributed = "false"
    run_curriculum = "all"
    n_rerun = 1
    num_workers = 1
    use_gpu = "true"
    pre_train = "false"
    use_pretrained_base, use_curriculum_base = "false", "false"
    pretrained_model, curriculum_model = None, None
    positive_class_weight, global_positive_class_weight = 1.0, 1.0
    n_examples, experiment_desc = None, None
    randomize_target_data = "false"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size)
    parser.add_argument("--accumulate", type=int, help="Gradient accumulation steps.", default=accumulate)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=lr)
    parser.add_argument("--training_file_name", type=str, help="File name cointaining examples", default=training_file)
    parser.add_argument("--pre_training_file_name", type=str, help="File name cointaining pre-training examples",
                        default=pre_training_file)
    parser.add_argument("--device_num", type=int, help="CUDA device number", default=device_num)
    parser.add_argument("--out_dim", type=int, help="Output dim of Biaffine Layer", default=out_dim)
    parser.add_argument("--skip_training", type=str, help="Skip Training", default=skip_training)
    parser.add_argument("--run_inference", type=str, help="run_inference", default=run_inference)
    parser.add_argument("--sigmoid_threshold", type=float, help="sigmoid_threshold", default=sigmoid_threshold)
    parser.add_argument("--context_attention", type=str, help="perform MHA between current text and context",
                        default=context_attention)
    parser.add_argument("--increase_token_type", type=str, help="Increase token type to 2",
                        default=increase_token_type)
    parser.add_argument("--experiment_number", type=str, help="experiment_number", default=experiment_number)
    parser.add_argument("--debug_mode", type=str, help="debug_mode", default=debug_mode)
    parser.add_argument("--train_distributed", type=str, help="train_distributed", default=train_distributed)
    parser.add_argument("--early_stopping", type=int, help="early_stopping", default=early_stopping)
    parser.add_argument("--run_curriculum", type=str, help="run_curriculum", default=run_curriculum)
    parser.add_argument("--use_gpu", type=str, help="use_gpu", default=use_gpu)
    parser.add_argument("--n_rerun", type=int, help="n_rerun", default=n_rerun)
    parser.add_argument("--num_workers", type=int, help="num_workers", default=num_workers)
    parser.add_argument("--predict_contextual_relationship", type=str,
                        help="use contextual rel matrix for additional prediction",
                        default=predict_contextual_relationship)
    parser.add_argument("--increase_positional_embeddings", type=str, help="Increase pos embedding size to 2500",
                        default=increase_positional_embeddings)
    parser.add_argument("--pre_train", type=str, help="pre_train", default=pre_train)
    parser.add_argument("--use_pretrained_base", type=str, help="use_pretrained_base", default=use_pretrained_base)
    parser.add_argument("--pretrained_model", type=str, help="pretrained_model", default=pretrained_model)
    parser.add_argument("--use_curriculum_base", type=str, help="use_curriculum_base", default=use_curriculum_base)
    parser.add_argument("--curriculum_model", type=str, help="curriculum_model", default=curriculum_model)
    parser.add_argument("--positive_class_weight", type=float, help="positive_class_weight", default=positive_class_weight)
    parser.add_argument("--global_positive_class_weight", type=float, help="global_positive_class_weight",
                        default=global_positive_class_weight)
    parser.add_argument("--n_examples", type=int, help="n_examples", default=n_examples)
    parser.add_argument("--experiment_desc", type=str, help="experiment_desc")
    parser.add_argument("--randomize_target_data", type=str, help="randomize_target_data",
                        default=randomize_target_data)

    parser.add_argument("--loss_local_rel_head_wt", type=float, help="loss_local_rel_head_wt", default=0.0)
    parser.add_argument("--loss_local_rel_deprel_wt", type=float, help="loss_local_rel_deprel_wt", default=0.0)
    parser.add_argument("--loss_edu_type_wt", type=float, help="loss_edu_type_wt", default=0.0)
    parser.add_argument("--loss_global_rel_head_wt", type=float, help="loss_global_rel_head_wt", default=0.0)
    parser.add_argument("--loss_global_rel_deprel_wt", type=float, help="loss_global_rel_deprel_wt", default=0.0)
    parser.add_argument("--loss_ctx_rel_deprel_wt", type=float, help="loss_ctx_rel_deprel_wt", default=0.0)
    parser.add_argument("--loss_local_rel_head_app_wt", type=float, help="loss_local_rel_head_app_wt", default=0.0)
    parser.add_argument("--loss_segmentation_wt", type=float, help="loss_segmentation_wt", default=0.0)

    argv = parser.parse_args()

    """ Loading config file """
    configurations = {"in_dim": in_dim, "out_dim": argv.out_dim,
                      "n_local_classes": 3, "n_global_classes": 3, "n_context_classes": 3,
                      "n_token_classes": 2, "n_type_classes": 4,
                      "n_layers": 4,  # N layers to pool during training
                      "n_heads": 4,  # N MHA heads
                      "n_self_attn_layers": 1,  # self MHA layers of current text
                      "n_ctx_attn_layers": 2,  # self MHA layers of ctx
                      "self_mha": do_self_mha,
                      "predict_contextual_relationship": True if argv.predict_contextual_relationship == "true" else False,
                      "context_attention": True if argv.context_attention == "true" else False,
                      "predict_segments": p_segments,
                      "predict_edu_type": p_edu_type, "predict_local_relations": p_local_relations,
                      "predict_global_relations": p_global_relations,
                      "base_transformer": "roberta-base", "lr": argv.learning_rate, "batch_size": argv.batch_size,
                      "accumulation": argv.accumulate, "device_num": argv.device_num,
                      "sigmoid_threshold": argv.sigmoid_threshold,
                      "training_file": argv.training_file_name, "num_epochs": argv.num_epochs,
                      "increase_token_type": True if argv.increase_token_type == "true" else False,
                      "debug_mode": True if argv.debug_mode == "true" else False,
                      "run_curriculum": argv.run_curriculum, "n_rerun": argv.n_rerun,
                      "increase_positional_embeddings": True if argv.increase_positional_embeddings == "true" else False,
                      "num_workers": argv.num_workers,
                      "use_gpu": True if argv.use_gpu == "true" else False,
                      "pre_training_file": argv.pre_training_file_name,
                      "pre_train": True if argv.pre_train == "true" else False,
                      "pretrained_model": argv.pretrained_model,
                      "use_pretrained_base": True if argv.use_pretrained_base == "true" else False,
                      "curriculum_model": argv.curriculum_model,
                      "use_curriculum_base": True if argv.use_curriculum_base == "true" else False,
                      "positive_class_weight": argv.positive_class_weight,
                      "global_positive_class_weight": argv.global_positive_class_weight,
                      "n_examples": argv.n_examples, "experiment_desc": argv.experiment_desc,
                      "experiment_number": argv.experiment_number,
                      "loss_local_rel_head_wt": argv.loss_local_rel_head_wt,
                      "loss_local_rel_deprel_wt": argv.loss_local_rel_deprel_wt,
                      "loss_edu_type_wt": argv.loss_edu_type_wt,
                      "loss_global_rel_head_wt": argv.loss_global_rel_head_wt,
                      "loss_global_rel_deprel_wt": argv.loss_global_rel_deprel_wt,
                      "loss_ctx_rel_deprel_wt": argv.loss_ctx_rel_deprel_wt,
                      "loss_local_rel_head_app_wt": argv.loss_local_rel_head_app_wt,
                      "loss_segmentation_wt": argv.loss_segmentation_wt,
                      "early_stopping": argv.early_stopping,
                      "skip_training": True if argv.skip_training == "true" else False,
                      "run_inference": True if argv.run_inference == "true" else False,
                      "randomize_target_data": True if argv.randomize_target_data == "true" else False
                      }

    """ Loading experiment details if any """
    if configurations["experiment_number"] != "-1":
        config_dict = json.load(open("./config.json", "r"))
        if config_dict.get(configurations["experiment_number"], None) is not None:
            print("\n::::: Loading execution parameters from config file ! :::::\n")
            for k, v in config_dict[configurations["experiment_number"]].items():
                if v in ["true", "false"]:
                    v = True if v == "true" else False
                """ Change the util mapping for C_target_dataset """
                if k.startswith("utils"):
                    k = k.replace("utils:", "").strip()
                    utils.curriculum_prediction_mapping["C_target_dataset"][k] = v
                    print("Note: A config value changed a default config in the utils mapping.")
                configurations[k] = v

    configurations["train_distributed"] = True if argv.train_distributed == "true" or configurations[
        "num_workers"] > 2 else False

    """ Normalizing Loss interpolation weights """
    tot_wt, zero_wt = 0.0, []
    loss_keys = ["loss_local_rel_head_wt", "loss_local_rel_deprel_wt", "loss_edu_type_wt", "loss_global_rel_head_wt",
                 "loss_global_rel_deprel_wt", "loss_ctx_rel_deprel_wt", "loss_local_rel_head_app_wt",
                 "loss_segmentation_wt"]
    for k in loss_keys:
        if configurations[k] == 0.0:
            zero_wt.append(k)
        else:
            tot_wt += configurations[k]
    remainder_wt = max(1.0 - tot_wt, 0.0)
    if len(zero_wt) > 0 and remainder_wt > 0:
        for k in zero_wt:
            configurations[k] = remainder_wt/len(zero_wt)

    print("\n:::::: LOSS INTERPOLATION VALUES:", {k: configurations[k] for k in loss_keys}, "::::::\n")

    if configurations["train_distributed"]:
        configurations["device_num"], device_num = int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_RANK"])
        print("\n====device_num:", device_num, "====\n")

    """ Creating folder structure """
    result_folder = path_prefix + "results/" + str(configurations["experiment_number"])
    if not os.path.isdir(result_folder):
        print("Creating New Directory: ", result_folder)
        os.mkdir(result_folder)

    if configurations["experiment_desc"] is None:
        raise Exception("Please provide a description of the run. Don't be Lazy!")
    else:
        configurations["experiment_desc"] += "_"+run_time
    if not os.path.isdir(result_folder + "/" + configurations["experiment_desc"]):
        print("Creating New Directory: ", result_folder + "/" + configurations["experiment_desc"])
        os.mkdir(result_folder + "/" + configurations["experiment_desc"])
        os.mkdir(result_folder + "/" + configurations["experiment_desc"] + "/models/")
        os.mkdir(result_folder + "/" + configurations["experiment_desc"] + "/results/")

    configurations["model_name"] = result_folder + "/" + configurations["experiment_desc"] + "/models/dialo_edu_ap_parser.pt"
    configurations["log_name"] = result_folder + "/" + configurations["experiment_desc"] + "/results/log.json"

    print("\n:::::EXECUTION CONFIGURATION:::::\n")
    print(configurations, "\n")
    execution_log = {"parameters": configurations, "training": {"losses": {}}, "inference_file": None}
    final_model_name = None

    if configurations["pre_train"]:
        torch.distributed.init_process_group(backend="nccl")
        train_filenames = ["imho_formatted_list_v1_train_list1.pkl", "imho_formatted_list_v1_train_list2.pkl",
                           "imho_formatted_list_v1_train_list3.pkl", "imho_formatted_list_v1_train_list4.pkl",
                           "imho_formatted_list_v1_train_list5.pkl", "imho_formatted_list_v1_train_list6.pkl",
                           "imho_formatted_list_v1_train_list7.pkl"]
        test_filename = "imho_formatted_list_v1_test.pkl"
        trainer = Trainer(configurations)
        best_valid_loss = float('inf')
        test_data_loader = trainer.get_pretrain_dataloader(path_prefix + "data/" + test_filename, train=False)
        for k, v in utils.curriculum_prediction_mapping["pretraining"].items():
            trainer.configuration[k] = v

        for epoch in range(configurations["num_epochs"]):
            print("======== PRE-TRAINING EPOCH:", epoch, "========\n")
            execution_log["training"]["losses"][epoch] = {}
            for ix, fname in enumerate(train_filenames):
                fnm = fname.replace(".pkl", "").strip()
                new_model_name = configurations["model_name"].replace(".pt", "_pretrain_file_"+fnm+"_epoch_"+str(epoch)+".pt")
                trainer.configuration["model_name"] = new_model_name
                train_data_loader = trainer.get_pretrain_dataloader(path_prefix + "data/" + fname, train=True)

                print("\n+++ Pre-Training on File,", fname, " +++\n")

                t_loss, v_loss, t_loss_local, v_loss_local, t_loss_global, v_loss_global, t_loss_local_lbl, v_loss_local_lbl, \
                    t_loss_global_lbl, v_loss_global_lbl, t_loss_span, v_loss_span, t_loss_typ, v_loss_typ, t_loss_ctx, \
                        v_loss_ctx, best_valid_loss, best_epoch, \
                            t_loss_local_app, v_loss_local_app, best_stats_list = run_training(1, trainer, best_valid_loss,
                                                                              new_model_name,
                                                                              configurations["early_stopping"],
                                                                              train_data_loader, test_data_loader,
                                                                              "pre-training",
                                                                              device_num)
                execution_log["training"]["losses"][epoch][fname] = {"t_loss": t_loss, "v_loss": v_loss,
                                                                     "t_loss_local": t_loss_local,
                                                                     "v_loss_local": v_loss_local,
                                                                     "t_loss_local_app": t_loss_local_app,
                                                                     "v_loss_local_app": v_loss_local_app,
                                                                     "t_loss_local_lbl": t_loss_local_lbl,
                                                                     "v_loss_local_lbl": v_loss_local_lbl,
                                                                     "t_loss_typ": t_loss_typ, "v_loss_typ": v_loss_typ,
                                                                     "best_valid_loss": best_valid_loss,
                                                                     "best_epoch": best_epoch,
                                                                     "best_stats_list": best_stats_list}
                if device_num == 0:
                    print("LOADING MODEL:", new_model_name)
                    state_dict = torch.load(new_model_name)
                    trainer.parser.load_state_dict(state_dict)
    else:
        if not configurations["skip_training"]:
            torch.distributed.init_process_group(backend="nccl")
            for rerun in range(configurations["n_rerun"]):
                torch.cuda.empty_cache()

                print("\n:::::: STARTING EXECUTION FOR RUN", rerun, "::::::\n")
                new_model_name = configurations["model_name"].replace(".pt", "_rerun_"+str(rerun)+".pt")
                configurations["model_name"] = new_model_name

                trainer = Trainer(configurations)
                print("\n:::::: Starting Training FOR RUN ", rerun, "::::::\n")

                for curr in trainer.curriculum_data_loader_list:
                    print("\n+++ Training Curriculum,", curr[0], " +++\n")

                    for k, v in utils.curriculum_prediction_mapping[curr[0]].items():
                        trainer.configuration[k] = v
                    print("\nUpdated Training Configuration as per curriculum: ", utils.curriculum_prediction_mapping[curr[0]], "\n")

                    best_valid_loss = float('inf')

                    t_loss, v_loss, t_loss_local, v_loss_local, t_loss_global, v_loss_global, t_loss_local_lbl, v_loss_local_lbl, \
                        t_loss_global_lbl, v_loss_global_lbl, t_loss_span, v_loss_span, t_loss_typ, v_loss_typ, t_loss_ctx, \
                            v_loss_ctx, best_valid_loss, best_epoch, \
                                t_loss_local_app, v_loss_local_app, best_stats_list = run_training(configurations["num_epochs"], trainer,
                                                                                  best_valid_loss, curr[1],
                                                                                  configurations["early_stopping"],
                                                                                  curr[-2], curr[-1], curr[0],
                                                                                  device_num)
                    print("Sleeping for some time !!! Zzzzz !!!")
                    time.sleep(60)
                    if os.path.exists(curr[1]):
                        print("Saved Model Path Found!! LOADING BEST CURRICULUM", curr[0], "MODEL")
                        state_dict = torch.load(curr[1])
                        trainer.parser.load_state_dict(state_dict)
                    else:
                        print("Saved Model Path Not Found !!")

                    if device_num == 0:
                        execution_log["training"]["losses"][curr[0]] = {"t_loss": t_loss, "v_loss": v_loss,
                                                                         "t_loss_local": t_loss_local,
                                                                         "v_loss_local": v_loss_local,
                                                                         "t_loss_local_app": t_loss_local_app,
                                                                         "v_loss_local_app": v_loss_local_app,
                                                                         "t_loss_global": t_loss_global,
                                                                         "v_loss_global": v_loss_global,
                                                                         "t_loss_local_lbl": t_loss_local_lbl,
                                                                         "v_loss_local_lbl": v_loss_local_lbl,
                                                                         "t_loss_global_lbl": t_loss_global_lbl,
                                                                         "v_loss_global_lbl": v_loss_global_lbl,
                                                                         "t_loss_span": t_loss_span, "v_loss_span": v_loss_span,
                                                                         "t_loss_typ": t_loss_typ, "v_loss_typ": v_loss_typ,
                                                                         "t_loss_ctx": t_loss_ctx, "v_loss_ctx": v_loss_ctx,
                                                                         "best_valid_loss": best_valid_loss,
                                                                         "best_epoch": best_epoch,
                                                                         "best_stats_list": best_stats_list}
                        final_model_name = curr[1]
                        """ Save stats file only if it's from the target datset """
                        if curr[0] == "C_target_dataset":
                            best_stats_list_df = pd.DataFrame(best_stats_list, columns=["task", "label", "metric", "score"])
                            best_stats_list_df["rerun"] = rerun
                            best_stats_list_df["experiment_no"] = configurations["experiment_number"]
                            best_stats_list_df["experiment_desc"] = configurations["experiment_desc"]
                            best_stats_list_df["timestamp"] = run_time

                            if os.path.exists(stats_file):
                                stats_df = pd.read_csv(stats_file)
                                best_stats_list_df = pd.concat([stats_df, best_stats_list_df])
                                best_stats_list_df = best_stats_list_df.reset_index(drop=True)
                            print("\n::: Saving stats file to ", stats_file, ":::\n")
                            best_stats_list_df.to_csv(stats_file, index=False)

                log_name = configurations["log_name"].replace(".json", "_" + str(rerun) + ".json")
                print("Saving Execution JSON to:", log_name)
                with open(log_name, 'w') as fp:
                    json.dump(execution_log, fp)

    if configurations["run_inference"] and final_model_name is not None:
        inferencer = Inference(configurations, final_model_name, True)
        inferencer.run_inference()
        print("\n::::::Running Inference::::::\n")

    if device_num == 0:
        print("ALL DONE!! GOING TO EXIT!!")
        os._exit(1)


if __name__ == '__main__':
    main()
