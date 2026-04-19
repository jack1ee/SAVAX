# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import time
import torch
from torch.nn.utils import clip_grad_norm_
# torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_sharing_strategy("file_system")
import os
import sys
import collections
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import dirname, abspath
import math

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)

from eval_utils import evaluate
import opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from misc.config_utils import merge_saved_options
from data.video_egome_dataset import PropSeqDataset, worker_init_fn
from SAVAX.SAVAX import build
from collections import OrderedDict

def apply_alternating_freeze(model, epoch, opt, logger):
    """
    Control the alternating freeze schedule.

    Returns:
        freeze_flags: dict indicating which modules are frozen.
        mode_name: current training mode for logging.
    """
    # Fall back to the default schedule before alternating training starts.
    if not getattr(opt, 'alt_train', False) or epoch < getattr(opt, 'alt_start_epoch', 5):
        return None, "Warmup/Standard"

    # Offset by start_epoch so the first alternating cycle starts at 0.
    freq = getattr(opt, 'alt_freq', 2)
    cycle_idx = ((epoch - opt.alt_start_epoch) // freq) % 2
    
    if cycle_idx == 0:
        # Mode A stabilizes downstream modules against the current sampler.
        mode_name = "Mode A (Train Downstream)"
        freeze_flags = {
            "sampler": True,
            "fusion": False,
            "view_enc": False,
            "backbone": False,
            "fast": False
        }
    else:
        # Mode B updates the sampler while keeping downstream heads fixed.
        mode_name = "Mode B (Train Sampler)"
        freeze_flags = {
            "sampler": False,
            "fusion": True,
            "view_enc": True,
            "backbone": True,
            "fast": True
        }
    
    apply_freeze_by_flags(model, freeze_flags)
    
    logger.info(f"[Alternating Strategy] Epoch {epoch}: Switched to {mode_name}")
    return freeze_flags, mode_name
def split_params_by_name(model):
    """
    Split parameters by name into coarse training buckets.

    Matching is substring-based and can be tightened here if needed.
    """
    groups = {"backbone": [], "sampler": [], "fusion": [], "view_enc": [], "fast_decay": [], "fast_nodecay": [], "decay": [], "nodecay": []}
    reduce_layer_name = "clr_mlp"  # High-LR layer tag.
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        name_lower = n.lower()
        is_norm_or_bias = (p.ndim == 1) or ('bias' in name_lower) or any(k in name_lower for k in ['layernorm','layer_norm','ln','norm','bn'])
        # First assign each parameter to a coarse module bucket.
        if 'sampler' in name_lower:
            bucket = "sampler"
        elif 'fuser' in name_lower:
            bucket = "fusion"
        elif 'view' in name_lower:     # Includes view_dict / view_embed / view_encoding.
            bucket = "view_enc"
        else:
            bucket = "backbone"
        # Split fast layers again by decay / no-decay.
        if reduce_layer_name in n:
            if is_norm_or_bias: groups["fast_nodecay"].append(p)
            else:               groups["fast_decay"].append(p)
        else:
            if is_norm_or_bias: groups["nodecay"].append(p)
            else:               groups["decay"].append(p)
        # Keep the module bucket for gradient clipping and freeze logic.
        groups[bucket].append(p)
    return groups

def build_param_groups(groups, base_lr, weight_decay, lr_mult):
    """
    Build optimizer param groups with per-group LR and decay settings.
    """
    pg = []
    # Standard backbone groups.
    if groups["decay"]:
        pg.append({"name":"decay", "params": groups["decay"], "lr": base_lr*lr_mult.get("backbone",1.0), "weight_decay": weight_decay})
    if groups["nodecay"]:
        pg.append({"name":"nodecay", "params": groups["nodecay"], "lr": base_lr*lr_mult.get("backbone",1.0), "weight_decay": 0.0})
    # Fast layers use a larger LR multiplier.
    if groups["fast_decay"]:
        pg.append({"name":"fast_decay", "params": groups["fast_decay"], "lr": base_lr*lr_mult.get("fast",10.0), "weight_decay": weight_decay})
    if groups["fast_nodecay"]:
        pg.append({"name":"fast_nodecay", "params": groups["fast_nodecay"], "lr": base_lr*lr_mult.get("fast",10.0), "weight_decay": 0.0})
    return pg

def apply_freeze_by_flags(model, flags):
    for n, m in model.named_parameters():
        name_lower = n.lower()
        if   'sampler' in name_lower:  m.requires_grad = not flags["sampler"]
        elif 'fuser'  in name_lower:  m.requires_grad = not flags["fusion"]
        elif 'view'    in name_lower:  m.requires_grad = not flags["view_enc"]
        else:                          m.requires_grad = not flags["backbone"]

def _tb_log_fine_pr_curves(eval_score, tf_writer, epoch, prefix='val'):
    """
    Plot PR curves returned by evaluation and log them to TensorBoard.

    Each tIoU gets one figure with correct/error curves and AP labels.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    tious = eval_score.get('iou_thres', [])
    pr_c_list = eval_score.get('fine_pr_curve_correct', [])
    pr_e_list = eval_score.get('fine_pr_curve_error', [])
    ap_c_list = eval_score.get('fine_auprc_correct', [])
    ap_e_list = eval_score.get('fine_auprc_error', [])

    if not isinstance(tious, (list, tuple)) or len(tious) == 0:
        return

    for i, tiou in enumerate(tious):
        prc = pr_c_list[i] if i < len(pr_c_list) else None  # {'precision': [...], 'recall': [...]}
        pre = pr_e_list[i] if i < len(pr_e_list) else None
        apc = float(ap_c_list[i]) if i < len(ap_c_list) else float('nan')
        ape = float(ap_e_list[i]) if i < len(ap_e_list) else float('nan')

        if prc is None and pre is None:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        if prc is not None:
            ax.plot(prc['recall'], prc['precision'], label=f'correct (AP={apc:.3f})')
        if pre is not None:
            ax.plot(pre['recall'], pre['precision'], label=f'error   (AP={ape:.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'Fine PR @ tIoU={tiou:.2f} | epoch {epoch}')
        ax.grid(True, alpha=0.3); ax.legend()

        tag = f'{prefix}/fine_pr_tiou_{str(tiou).replace(".","p")}'
        tf_writer.add_figure(tag, fig, global_step=epoch)
        plt.close(fig)

def _extract_overall_threshold(eval_score):
    """
    Read the overall threshold from either legacy or new evaluator keys.

    Returns:
        (threshold or None, key_used or None)
    """
    key_priority = ["threshold_error", "all_best_threshold_error_by_f1", "all_threshold_error"]
    for key in key_priority:
        v = eval_score.get(key, None)
        if isinstance(v, list) and len(v) > 0:
            return float(v[0]), key
        if isinstance(v, (int, float)):
            return float(v), key
    return None, None


def create_loader(dataset, batch_size, shuffle, nthreads):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": nthreads,
        "collate_fn": dataset.collate_fn,
        "pin_memory": True,
        "worker_init_fn": worker_init_fn,
    }
    if nthreads > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def move_batch_to_device(dt, device):
    dt = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in dt.items()}
    dt["video_target"] = [
        {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in video_info.items()}
        for video_info in dt["video_target"]
    ]
    return collections.defaultdict(lambda: None, dt)


def load_resume_state(opt, save_folder, logger):
    saved_info = {"best": {}, "last": {}, "history": {}, "eval_history": {}}
    checkpoint = None

    if not opt.start_from:
        return saved_info, checkpoint, 0, 0

    opt.pretrain = "none"
    info_path = os.path.join(save_folder, "info.json")
    with open(info_path, "r") as handle:
        logger.info("Load info from %s", info_path)
        saved_info = json.load(handle)

    mode_info = saved_info.get(opt.start_from_mode[:4], {})
    prev_opt = mode_info.get("opt", {})
    excluded_keys = {
        "cfg_path",
        "id",
        "gpu_id",
        "device",
        "disable_tqdm",
        "disable_cudnn",
        "debug",
        "seed",
        "random_seed",
        "start_from",
        "start_from_mode",
        "pretrain",
        "pretrain_path",
        "save_dir",
    }
    merge_saved_options(opt, prev_opt, excluded_keys=excluded_keys, logger=logger)

    checkpoint_name = "model-best.pth" if opt.start_from_mode == "best" else "model-last.pth"
    checkpoint_path = os.path.join(save_folder, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(opt.device))

    start_epoch = int(mode_info.get("epoch", -1)) + 1
    iteration = int(mode_info.get("iter", 0))
    logger.info("Loading pth from %s, iteration:%s", checkpoint_path, iteration)
    return saved_info, checkpoint, start_epoch, iteration


def load_pretrained_weights(model, opt, logger):
    if opt.start_from or opt.pretrain == "none":
        return

    logger.info("Load pre-trained parameters from %s", opt.pretrain_path)
    model_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
    if opt.pretrain == "encoder":
        encoder_filter = model.get_filter_rule_for_encoder()
        encoder_pth = {k: v for k, v in model_pth["model"].items() if encoder_filter(k)}
        model.load_state_dict(encoder_pth, strict=True)
    elif opt.pretrain == "decoder":
        encoder_filter = model.get_filter_rule_for_encoder()
        decoder_pth = {k: v for k, v in model_pth["model"].items() if not encoder_filter(k)}
        model.load_state_dict(decoder_pth, strict=True)
    elif opt.pretrain == "full":
        matching_weights = {}
        for layer_name, params in model_pth["model"].items():
            if layer_name in model.state_dict() and model.state_dict()[layer_name].size() == params.size():
                matching_weights[layer_name] = params
        model.load_state_dict(matching_weights, strict=False)
    else:
        raise ValueError("wrong value of opt.pretrain")


def build_optimizer_and_scheduler(model, opt, train_loader, checkpoint, iteration, logger):
    base_lr = opt.lr
    weight_decay = opt.weight_decay
    clip_values = {
        "backbone": getattr(opt, "grad_clip_backbone", opt.grad_clip),
        "sampler": getattr(opt, "grad_clip_sampler", 0.5),
        "fusion": getattr(opt, "grad_clip_fusion", 1.0),
        "view_enc": getattr(opt, "grad_clip_view", 1.0),
    }

    default_freeze_flags = {"sampler": False, "fusion": False, "view_enc": False, "backbone": False}
    apply_freeze_by_flags(model, default_freeze_flags)

    name_groups = split_params_by_name(model)
    lr_mult = {"backbone": 1.0, "fast": 10.0}
    param_groups = build_param_groups(name_groups, base_lr, weight_decay, lr_mult)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=getattr(opt, "betas", (0.9, 0.999)),
        eps=getattr(opt, "adam_eps", 1e-8),
    )

    total_steps = max(1, opt.epoch * len(train_loader))
    warmup_steps = getattr(opt, "warmup_steps", int(getattr(opt, "warmup_ratio", 0.05) * total_steps))
    min_lr_ratio = getattr(opt, "min_lr_ratio", 0.0)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    if checkpoint is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Loaded optimizer state from checkpoint.")
        except Exception as exc:
            logger.warning("Skip loading previous optimizer state: %s", exc)
    if iteration > 0:
        lr_scheduler.step(iteration)

    opt.current_lr = optimizer.param_groups[0]["lr"]
    return optimizer, lr_scheduler, name_groups, clip_values, default_freeze_flags


def run_validation(model, criterion, postprocessors, val_loader, val_dataset, save_folder, logger, tf_writer, opt, epoch, iteration):
    result_json_path = os.path.join(save_folder, "prediction", "num{}_epoch{}.json".format(len(val_dataset), epoch))
    eval_score, eval_loss = evaluate(
        model,
        criterion,
        postprocessors,
        val_loader,
        result_json_path,
        logger=logger,
        alpha=opt.ec_alpha,
        device=opt.device,
        debug=opt.debug,
        w_correct=opt.w_correct,
        eval_phase="val",
        fix_fine_thresh=opt.fix_fine_thresh,
        num_thresh=opt.num_thresh,
        thresh_sampling=opt.thresh_sampling,
    )
    _tb_log_fine_pr_curves(eval_score, tf_writer, epoch, prefix="val")

    if opt.caption_decoder_type == "none":
        current_score = 2.0 / (1.0 / eval_score["Precision"] + 1.0 / eval_score["Recall"])
    elif opt.criteria_for_best_ckpt == "dvc":
        current_score = np.array(eval_score["METEOR"]).mean() + np.array(eval_score["SODA_Meteor"]).mean()
    elif opt.criteria_for_best_ckpt == "mimic":
        current_score = np.array(eval_score["fine_auprc_error_mean"]).mean() + 0.2 * np.array(eval_score["all_auprc_error"]).mean()
    else:
        current_score = (
            np.array(eval_score["para_METEOR"]).mean()
            + np.array(eval_score["para_CIDEr"]).mean()
            + np.array(eval_score["para_Bleu_4"]).mean()
        )

    for key, val in eval_score.items():
        try:
            tf_writer.add_scalar(key, float(np.nanmean(np.array(val, dtype=np.float64))), iteration)
        except Exception:
            pass
    for loss_type, loss_value in eval_loss.items():
        tf_writer.add_scalar("eval_" + loss_type, loss_value, iteration)

    print_info = "\n".join(
        [key + ":" + str(eval_score[key]) for key in eval_score.keys() if key not in ["fine_pr_curve_correct", "fine_pr_curve_error"]]
    )
    logger.info("\nValidation results of iter {}:\n{}".format(iteration, print_info))
    logger.info("\noverall score of iter {}: {}\n".format(iteration, current_score))
    return current_score, eval_score, eval_loss, result_json_path


def run_train_epoch(model, criterion, train_loader, optimizer, lr_scheduler, opt, logger, tf_writer,
                    epoch, iteration, name_groups, clip_values, weight_dict, loss_history, lr_history):
    loss_sum = OrderedDict()
    view_acc_list = []
    bad_video_num = 0
    start_time = time.time()

    for dt in tqdm(train_loader, disable=opt.disable_tqdm):
        if opt.device == "cuda":
            torch.cuda.synchronize(opt.device)
        if opt.debug and (iteration + 1) % 5 == 0:
            iteration += 1
            break

        iteration += 1
        optimizer.zero_grad()
        dt = move_batch_to_device(dt, opt.device)

        output, loss = model(dt, criterion, opt.transformer_input_type)
        final_loss = sum(loss[key] * weight_dict[key] for key in loss.keys() if key in weight_dict)
        if not final_loss.requires_grad:
            raise RuntimeError("final_loss has no gradient information; check loss weights and frozen parameters")

        final_loss.backward()
        clip_grad_norm_(name_groups["backbone"], clip_values["backbone"])
        clip_grad_norm_(name_groups["sampler"], clip_values["sampler"])
        clip_grad_norm_(name_groups["fusion"], clip_values["fusion"])
        clip_grad_norm_(name_groups["view_enc"], clip_values["view_enc"])
        optimizer.step()
        lr_scheduler.step()

        opt.current_lr = optimizer.param_groups[0]["lr"]
        for loss_key, loss_value in loss.items():
            loss_sum[loss_key] = loss_sum.get(loss_key, 0) + loss_value.item()
        loss_sum["total_loss"] = loss_sum.get("total_loss", 0) + final_loss.item()
        if "view_acc" in output:
            view_acc_list.append(output["view_acc"])

        if opt.device == "cuda":
            torch.cuda.synchronize()

        losses_log_every = max(1, int(len(train_loader) / 5))
        if opt.debug:
            losses_log_every = 6

        if iteration % losses_log_every == 0:
            end_time = time.time()
            averaged_loss = OrderedDict((key, np.round(value / losses_log_every, 3).item()) for key, value in loss_sum.items())
            train_log = "ID {} iter {} (epoch {}), \nloss = {}, \ntime/iter = {:.3f}, bad_vid = {:.3f}".format(
                opt.id, iteration, epoch, averaged_loss, (end_time - start_time) / losses_log_every, bad_video_num
            )
            if view_acc_list:
                train_log += ", view_acc = {:.3f}".format(np.mean(view_acc_list))
                view_acc_list = []
            logger.info(train_log)

            tf_writer.add_scalar("lr", opt.current_lr, iteration)
            for loss_type, loss_value in averaged_loss.items():
                tf_writer.add_scalar(loss_type, loss_value, iteration)
            loss_history[iteration] = averaged_loss
            lr_history[iteration] = opt.current_lr

            loss_sum = OrderedDict()
            start_time = time.time()
            bad_video_num = 0
            torch.cuda.empty_cache()
    return iteration
        
def train(opt):
    set_seed(opt.seed)
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, "train.log")
    tf_writer = SummaryWriter(os.path.join(save_folder, "tf_summary"))

    if not opt.start_from:
        backup_envir(save_folder)
        logger.info("backup evironment completed !")

    saved_info, checkpoint, epoch, iteration = load_resume_state(opt, save_folder, logger)
    resume_history = saved_info.get("history", {})
    best_val_score = saved_info.get(opt.start_from_mode[:4], {}).get("best_val_score", -1e5)
    val_result_history = resume_history.get("val_result_history", {})
    loss_history = resume_history.get("loss_history", {})
    lr_history = resume_history.get("lr_history", {})

    train_dataset = PropSeqDataset(
        [opt.ego_train_caption_file, opt.exo_train_caption_file],
        getattr(opt, "visual_feature_folder", None),
        opt.dict_file,
        True,
        "gt",
        opt,
    )
    val_dataset = PropSeqDataset(
        [opt.ego_val_caption_file, opt.exo_val_caption_file],
        getattr(opt, "visual_feature_folder", None),
        opt.dict_file,
        False,
        "gt",
        opt,
    )
    train_loader = create_loader(train_dataset, opt.batch_size, True, opt.nthreads)
    val_loader = create_loader(val_dataset, opt.batch_size_for_eval, False, opt.nthreads)

    model, criterion, postprocessors = build(opt)
    model.translator = train_dataset.translator
    model.train()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    else:
        load_pretrained_weights(model, opt, logger)

    model.to(opt.device)
    optimizer, lr_scheduler, name_groups, clip_values, default_freeze_flags = build_optimizer_and_scheduler(
        model, opt, train_loader, checkpoint, iteration, logger
    )

    print_opt(opt, model, logger)
    print_alert_message("Strat training !", logger)

    weight_dict = criterion.weight_dict
    logger.info("loss type: %s", weight_dict.keys())
    logger.info("loss weights: %s", weight_dict.values())

    es_cnt = 0
    while epoch < opt.epoch:
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(
                opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                opt.scheduled_sampling_max_prob,
            )
            model.caption_head.ss_prob = opt.ss_prob

        logger.info("lr:%s", float(opt.current_lr))
        alt_flags, _ = apply_alternating_freeze(model, epoch, opt, logger)
        freeze_flags = alt_flags if alt_flags is not None else default_freeze_flags.copy()
        if alt_flags is None:
            apply_freeze_by_flags(model, freeze_flags)

        (model.module if hasattr(model, "module") else model).apply_view_schedules(epoch)
        iteration = run_train_epoch(
            model, criterion, train_loader, optimizer, lr_scheduler, opt, logger, tf_writer,
            epoch, iteration, name_groups, clip_values, weight_dict, loss_history, lr_history
        )

        if epoch % opt.save_checkpoint_every == 0 and epoch >= opt.min_epoch_when_save:
            saved_pth = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
            checkpoint_path = os.path.join(save_folder, "model_iter_{}.pth".format(iteration) if opt.save_all_checkpoint else "model-last.pth")
            torch.save(saved_pth, checkpoint_path)

            model.eval()
            current_score, eval_score, eval_loss, result_json_path = run_validation(
                model, criterion, postprocessors, val_loader, val_dataset, save_folder, logger, tf_writer, opt, epoch, iteration
            )
            val_result_history[epoch] = {"eval_score": eval_score}
            logger.info("Save model at iter {} / epoch {} to {}.".format(iteration, epoch, checkpoint_path))

            if current_score >= best_val_score:
                best_val_score = current_score
                es_cnt = 0
                fine_th = None
                if isinstance(eval_score.get("thres_fine_global"), list) and eval_score["thres_fine_global"]:
                    fine_th = float(eval_score["thres_fine_global"][0])
                overall_th, overall_key = _extract_overall_threshold(eval_score)
                if overall_th is not None:
                    logger.info("[threshold] using overall threshold %.6f from eval_score['%s']", overall_th, overall_key)
                else:
                    logger.info("[threshold] no overall threshold key found in eval_score")

                saved_info["best"] = {
                    "opt": dict(vars(opt)),
                    "iter": iteration,
                    "epoch": epoch,
                    "best_val_score": best_val_score,
                    "result_json_path": result_json_path,
                    "avg_proposal_num": eval_score["avg_proposal_number"],
                    "Precision": eval_score["Precision"],
                    "Recall": eval_score["Recall"],
                }
                saved_info["best"]["opt"]["fixed_fine_threshold"] = fine_th
                saved_info["best"]["opt"]["fixed_overall_threshold"] = overall_th
                torch.save(saved_pth, os.path.join(save_folder, "model-best.pth"))
                logger.info("Save Best-model at iter {} / epoch {} to checkpoint file.".format(iteration, epoch))
            else:
                es_cnt += 1

            saved_info["last"] = {
                "opt": dict(vars(opt)),
                "iter": iteration,
                "epoch": epoch,
                "best_val_score": best_val_score,
            }
            saved_info["history"] = {
                "val_result_history": val_result_history,
                "loss_history": loss_history,
                "lr_history": lr_history,
            }
            with open(os.path.join(save_folder, "info.json"), "w") as handle:
                json.dump(saved_info, handle)
            logger.info("Save info to info.json")
            model.train()

        epoch += 1
        opt.current_lr = optimizer.param_groups[0]["lr"]
        torch.cuda.empty_cache()
        if es_cnt > opt.max_es_cnt:
            logger.info("Early stop at {} with val score {}".format(epoch, best_val_score))
            break

    tf_writer.close()

    return saved_info


if __name__ == '__main__':
    opt = opts.parse_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    train(opt)
