from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import pprint
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.video_egome_dataset import PropSeqDataset, worker_init_fn
from eval_utils import evaluate
from misc.config_utils import apply_cli_overrides, load_json_file, load_yaml_config, make_namespace, merge_saved_options, validate_eval_config
from misc.utils import create_logger
from SAVAX.SAVAX import build


def build_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True, help="config file")
    parser.add_argument("--id", type=str, default=None, help="run id used in output filenames")
    parser.add_argument("--eval_save_dir", type=str, default=None)
    parser.add_argument("--eval_folder", type=str, default=None)
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--eval_tool_version", type=str, default=None, choices=["2018", "2021"])
    parser.add_argument("--eval_proposal_type", type=str, default=None)
    parser.add_argument("--eval_transformer_input_type", type=str, default=None, choices=["gt_proposals", "queries"])
    parser.add_argument("--eval_video_feature_folder", type=str, nargs="+", default=None)
    parser.add_argument("--eval_visual_feature_type", type=str, nargs="+", default=None)
    parser.add_argument("--gpu_id", type=str, nargs="+", default=None)
    parser.add_argument("--eval_device", type=str, default=None)
    parser.add_argument("--re_eval", action="store_true", default=None)
    parser.add_argument("--ec_alpha", type=float, default=None)
    return parser


def parse_eval_opts():
    parser = build_eval_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cfg_path", type=str, required=True)
    pre_args, _ = pre_parser.parse_known_args()

    config = load_yaml_config(pre_args.cfg_path)
    cli_args = parser.parse_args()
    overrides = {key: value for key, value in vars(cli_args).items() if key != "cfg_path"}
    config = apply_cli_overrides(config, overrides)
    config["cfg_path"] = cli_args.cfg_path
    validate_eval_config(config)
    return make_namespace(config)


def create_loader(dataset, batch_size, nthreads):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": nthreads,
        "collate_fn": dataset.collate_fn,
        "pin_memory": True,
        "worker_init_fn": worker_init_fn,
    }
    if nthreads > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def resolve_eval_folder(opt):
    if opt.eval_folder.strip("./").startswith(opt.eval_save_dir):
        folder_path = opt.eval_folder
    else:
        folder_path = os.path.join(opt.eval_save_dir, opt.eval_folder)
    if "*" in folder_path:
        return glob.glob(folder_path)
    return [folder_path]


def restore_training_options(opt, info_path, logger):
    if not os.path.exists(info_path):
        logger.warning("info.json not found at %s; evaluation will use only YAML options.", info_path)
        return

    saved_info = load_json_file(info_path)
    saved_opt = saved_info.get("best", {}).get("opt", {})
    excluded_keys = {
        "cfg_path",
        "id",
        "gpu_id",
        "eval_save_dir",
        "eval_folder",
        "eval_model_path",
        "eval_tool_version",
        "eval_proposal_type",
        "eval_transformer_input_type",
        "eval_video_feature_folder",
        "eval_visual_feature_type",
        "eval_device",
        "re_eval",
        "ec_alpha",
    }
    merge_saved_options(opt, saved_opt, excluded_keys=excluded_keys, logger=logger)


def build_output_path(folder_path, opt, epoch, dataset_size):
    date_prefix = time.strftime("%y%m%d-%H%M%S_", time.localtime())
    file_name = "{}{}_epoch{}_num{}_alpha{}.json".format(date_prefix, str(opt.id), epoch, dataset_size, opt.ec_alpha)
    if opt.eval_transformer_input_type is not None and "gt" in opt.eval_transformer_input_type:
        file_name = "{}{}_gt_prop_epoch{}_num{}_alpha{}.json".format(date_prefix, str(opt.id), epoch, dataset_size, opt.ec_alpha)
    return os.path.join(folder_path, file_name)


def main(opt):
    folder_path_list = resolve_eval_folder(opt)
    json_name = os.path.basename(opt.ego_eval_caption_file).split(".")[0]
    model_dir = Path(opt.eval_model_path).parts[-2] if opt.eval_model_path else "default"
    save_log_filename = "eval_{}_{}.log".format(json_name, model_dir.split("assessment")[-1])
    if opt.eval_transformer_input_type is not None and "gt" in opt.eval_transformer_input_type:
        save_log_filename = "eval_{}_gt_prop.log".format(json_name)

    for folder_path in folder_path_list:
        Path(folder_path).mkdir(exist_ok=True, parents=True)
        if not opt.re_eval and os.path.exists(os.path.join(folder_path, save_log_filename)):
            print("skip: eval file exists", folder_path)
            continue

        logger = create_logger(folder_path, save_log_filename)
        logger.info("logger created in %s/%s", folder_path, save_log_filename)

        model_path = opt.eval_model_path or os.path.join(folder_path, "model-best.pth")
        info_path = os.path.join(os.path.dirname(model_path), "info.json")
        restore_training_options(opt, info_path, logger)

        if opt.eval_transformer_input_type is not None:
            opt.transformer_input_type = opt.eval_transformer_input_type
        if not torch.cuda.is_available():
            opt.nthreads = 0
        if opt.eval_video_feature_folder is not None:
            opt.visual_feature_folder = opt.eval_video_feature_folder
        if opt.eval_visual_feature_type is not None:
            opt.visual_feature_type = opt.eval_visual_feature_type

        logger.info(vars(opt))
        dataset = PropSeqDataset(
            [opt.ego_eval_caption_file, opt.exo_eval_caption_file],
            getattr(opt, "visual_feature_folder", None),
            opt.dict_file,
            False,
            opt.eval_proposal_type,
            opt,
        )
        loader = create_loader(dataset, opt.batch_size_for_eval, opt.nthreads)

        model, criterion, postprocessors = build(opt)
        model.translator = dataset.translator

        if not os.path.exists(model_path):
            raise AssertionError("File {} does not exist".format(model_path))

        logger.debug("Loading model from %s", model_path)
        checkpoint = torch.load(model_path, map_location=opt.eval_device)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        model.to(opt.eval_device)

        out_json_path = build_output_path(folder_path, opt, epoch, len(loader.dataset))
        caption_scores, eval_loss = evaluate(
            model,
            criterion,
            postprocessors,
            loader,
            out_json_path,
            logger,
            alpha=opt.ec_alpha,
            dvc_eval_version=opt.eval_tool_version,
            device=opt.eval_device,
            debug=False,
            skip_lang_eval=False,
            w_correct=opt.w_correct,
            eval_phase="test",
            fix_fine_thresh=opt.fix_fine_thresh,
            num_thresh=opt.num_thresh,
            thresh_sampling=opt.thresh_sampling,
            fixed_fine_thresh=getattr(opt, "fixed_fine_threshold", None),
            fixed_overall_thresh=getattr(opt, "fixed_overall_threshold", None),
            vis_out_dir=getattr(opt, "vis_out_dir", None),
            vis_num=getattr(opt, "vis_num", 0),
            vis_logit_norm=getattr(opt, "vis_logit_norm", "none"),
            vis_keys=getattr(opt, "vis_keys", "none"),
        )

        formatted_iou_thre = pprint.pformat(caption_scores["iou_thres"])
        formatted_avg_eval_score = "\n".join(
            [key + ":" + str(caption_scores[key]) for key in caption_scores.keys() if key not in ["fine_pr_curve_correct", "fine_pr_curve_error"]]
        )
        logger.info(
            "\nValidation result based on all {} val videos:\n scores:\n{}\n"
            "iou_threshold:\n{}".format(len(loader.dataset), formatted_avg_eval_score, formatted_iou_thre)
        )
        logger.info("saving results json to %s", out_json_path)


if __name__ == "__main__":
    opt = parse_eval_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in opt.gpu_id])
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    if getattr(opt, "disable_cudnn", False):
        torch.backends.cudnn.enabled = False
    main(opt)
