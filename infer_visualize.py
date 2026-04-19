from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import logging
import os
import re
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM")


def load_runtime_modules():
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from data.video_egome_dataset import PropSeqDataset, worker_init_fn
    from misc.config_utils import (
        apply_cli_overrides,
        load_json_file,
        load_yaml_config,
        make_namespace,
        merge_saved_options,
        validate_eval_config,
    )
    from SAVAX.SAVAX import build

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "tqdm": tqdm,
        "PropSeqDataset": PropSeqDataset,
        "worker_init_fn": worker_init_fn,
        "apply_cli_overrides": apply_cli_overrides,
        "load_json_file": load_json_file,
        "load_yaml_config": load_yaml_config,
        "make_namespace": make_namespace,
        "merge_saved_options": merge_saved_options,
        "validate_eval_config": validate_eval_config,
        "build": build,
    }


class DatasetView(object):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.collate_fn = base_dataset.collate_fn
        self.opt = base_dataset.opt
        self.translator = base_dataset.translator

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]

    def _lazy_open_env(self):
        self.base_dataset._lazy_open_env()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run SAVAX inference on one or more paired EgoMe samples and render ego-video visualizations "
            "with predicted DVC captions and step correctness."
        )
    )
    parser.add_argument("--cfg_path", type=str, required=True, help="Evaluation YAML config.")
    parser.add_argument("--id", type=str, default=None, help="Optional run id suffix for outputs.")
    parser.add_argument("--eval_model_path", type=str, default=None, help="Checkpoint path. Defaults to YAML.")
    parser.add_argument("--eval_device", type=str, default=None, help="Inference device, e.g. cpu or cuda.")
    parser.add_argument("--gpu_id", type=str, nargs="+", default=None, help="Visible GPU ids.")
    parser.add_argument("--batch_size_for_eval", type=int, default=None, help="Inference batch size override.")
    parser.add_argument("--nthreads", type=int, default=None, help="DataLoader workers override.")

    parser.add_argument("--sample_key", type=str, nargs="+", default=None, help="Sample ids to visualize.")
    parser.add_argument(
        "--sample_keys_file",
        type=str,
        default=None,
        help="Text file with one sample id per line; merged with --sample_key.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum number of samples to render after filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--match_policy",
        type=str,
        default="intersection",
        choices=["annotation", "ego_only", "intersection"],
        help=(
            "How batch mode selects samples when directories are provided: "
            "'annotation' uses only annotation/filter selection, "
            "'ego_only' requires only ego videos to exist, "
            "'intersection' requires both ego and exo matches when exo dir is set."
        ),
    )

    parser.add_argument("--ego_video", type=str, default=None, help="Explicit ego video path for single-sample mode.")
    parser.add_argument("--exo_video", type=str, default=None, help="Explicit exo video path for single-sample mode.")
    parser.add_argument("--ego_video_dir", type=str, default=None, help="Directory containing raw ego videos.")
    parser.add_argument("--exo_video_dir", type=str, default=None, help="Directory containing raw exo videos.")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write rendered videos and prediction JSON.",
    )
    parser.add_argument(
        "--output_video_suffix",
        type=str,
        default="_pred.mp4",
        help="Filename suffix for rendered ego videos.",
    )
    parser.add_argument(
        "--proposal_score_threshold",
        type=float,
        default=0.0,
        help="Filter low-confidence proposals before rendering.",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=0,
        help="Maximum number of predicted steps to visualize per sample. 0 means use model pred_event_count.",
    )
    parser.add_argument(
        "--fine_threshold",
        type=float,
        default=None,
        help="Threshold for classifying a step as CORRECT vs ERROR. Defaults to fixed_fine_threshold or 0.5.",
    )
    parser.add_argument(
        "--overall_threshold",
        type=float,
        default=None,
        help="Threshold for classifying the whole video as CORRECT vs ERROR. Defaults to fixed_overall_threshold or 0.5.",
    )
    parser.add_argument(
        "--render_fps",
        type=float,
        default=0.0,
        help="Override output fps. 0 means use source ego video fps.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Render only the first N frames for quick debugging. 0 means full video.",
    )
    parser.add_argument(
        "--panel_ratio",
        type=float,
        default=0.24,
        help="Bottom caption panel height as a fraction of frame height.",
    )
    return parser


def parse_opts():
    parser = build_parser()
    cli_args = parser.parse_args()
    runtime = load_runtime_modules()

    config = runtime["load_yaml_config"](cli_args.cfg_path)
    overrides = {
        key: value
        for key, value in vars(cli_args).items()
        if key in {
            "id",
            "eval_model_path",
            "eval_device",
            "gpu_id",
            "batch_size_for_eval",
            "nthreads",
        }
    }
    config = runtime["apply_cli_overrides"](config, overrides)
    config["cfg_path"] = cli_args.cfg_path
    runtime["validate_eval_config"](config)
    opt = runtime["make_namespace"](config)
    for key, value in vars(cli_args).items():
        if value is None and hasattr(opt, key):
            continue
        setattr(opt, key, value)
    return opt


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger("infer_visualize")


def create_loader(dataset, batch_size, nthreads):
    runtime = load_runtime_modules()
    batch_size = batch_size or getattr(dataset.opt, "batch_size_for_eval", None) or 1
    nthreads = 0 if nthreads is None else nthreads
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": nthreads,
        "collate_fn": dataset.collate_fn,
        "pin_memory": True,
        "worker_init_fn": runtime["worker_init_fn"],
    }
    if nthreads > 0:
        loader_kwargs["prefetch_factor"] = 2
    return runtime["DataLoader"](dataset, **loader_kwargs)


def restore_training_options(opt, info_path, logger):
    runtime = load_runtime_modules()
    if not os.path.exists(info_path):
        logger.warning("info.json not found at %s; using YAML options only.", info_path)
        return

    saved_info = runtime["load_json_file"](info_path)
    saved_opt = saved_info.get("best", {}).get("opt", {})
    excluded_keys = {
        "cfg_path",
        "id",
        "gpu_id",
        "eval_model_path",
        "eval_device",
        "batch_size_for_eval",
        "nthreads",
    }
    runtime["merge_saved_options"](opt, saved_opt, excluded_keys=excluded_keys, logger=None)


def load_requested_keys(opt):
    requested = []
    if opt.sample_key:
        requested.extend(opt.sample_key)
    if opt.sample_keys_file:
        with open(opt.sample_keys_file, "r", encoding="utf-8") as handle:
            requested.extend([line.strip() for line in handle if line.strip()])
    return list(dict.fromkeys(requested))


def build_video_index(root_dir):
    if root_dir is None:
        return {}
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError("video directory does not exist: {}".format(root))

    index = collections.defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in VIDEO_EXTENSIONS:
            continue
        index[path.stem].append(path)
        index[path.name].append(path)
    return index


def choose_first_path(index, candidates):
    for candidate in candidates:
        if not candidate:
            continue
        paths = index.get(candidate)
        if paths:
            return str(sorted(paths)[0])
    return None


def build_video_candidates(sample_id, anno_key, anno_info):
    candidates = [sample_id, anno_key]
    source_video = anno_info.get("source_video")
    if isinstance(source_video, str) and source_video:
        source_path = Path(source_video)
        candidates.extend([source_video, source_path.name, source_path.stem])
    match_video = anno_info.get("match_video")
    if isinstance(match_video, str) and match_video:
        match_path = Path(match_video)
        candidates.extend([match_video, match_path.name, match_path.stem])
    deduped = []
    seen = set()
    for item in candidates:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def resolve_sample_videos(dataset, sample_id, ego_index, exo_index, opt):
    ego_key = dataset.id2ego[sample_id]
    exo_key = dataset.id2exo[sample_id]
    ego_info = dataset.anno_ego[ego_key]
    exo_info = dataset.anno_exo[exo_key]

    ego_candidates = build_video_candidates(sample_id, ego_key, ego_info)
    exo_candidates = build_video_candidates(sample_id, exo_key, exo_info)

    ego_path = opt.ego_video if opt.ego_video and len(load_requested_keys(opt)) <= 1 else choose_first_path(ego_index, ego_candidates)
    exo_path = opt.exo_video if opt.exo_video and len(load_requested_keys(opt)) <= 1 else choose_first_path(exo_index, exo_candidates)

    return {
        "sample_id": sample_id,
        "ego_key": ego_key,
        "exo_key": exo_key,
        "ego_video_path": ego_path,
        "exo_video_path": exo_path,
        "ego_candidates": ego_candidates,
        "exo_candidates": exo_candidates,
    }


def select_sample_indices(dataset, opt, logger):
    requested_keys = load_requested_keys(opt)
    selected_ids = requested_keys if requested_keys else list(dataset.keys)

    unknown_ids = [sample_id for sample_id in selected_ids if sample_id not in dataset.id2ego]
    if unknown_ids:
        raise KeyError("Unknown sample ids: {}".format(unknown_ids[:10]))

    ego_index = build_video_index(opt.ego_video_dir) if opt.ego_video_dir else {}
    exo_index = build_video_index(opt.exo_video_dir) if opt.exo_video_dir else {}

    sample_infos = []
    for sample_id in selected_ids:
        info = resolve_sample_videos(dataset, sample_id, ego_index, exo_index, opt)
        use_sample = True
        if opt.match_policy == "ego_only":
            use_sample = info["ego_video_path"] is not None or not opt.ego_video_dir
        elif opt.match_policy == "intersection":
            use_sample = info["ego_video_path"] is not None or not opt.ego_video_dir
            if opt.exo_video_dir:
                use_sample = use_sample and info["exo_video_path"] is not None
        if use_sample:
            sample_infos.append(info)

    if opt.max_samples and opt.max_samples > 0:
        sample_infos = sample_infos[: opt.max_samples]

    if not sample_infos:
        raise RuntimeError("No samples selected after filtering.")

    missing_ego = [item["sample_id"] for item in sample_infos if item["ego_video_path"] is None]
    if missing_ego:
        logger.warning("Selected %d samples without resolved ego videos. They will fail at render time.", len(missing_ego))

    indices = [dataset.keys.index(item["sample_id"]) for item in sample_infos]
    return indices, {item["sample_id"]: item for item in sample_infos}


def predict_selected(model, criterion, postprocessors, loader, device, transformer_input_type, score_threshold):
    runtime = load_runtime_modules()
    torch = runtime["torch"]
    tqdm = runtime["tqdm"]
    predictions = {}
    with torch.no_grad():
        for dt in tqdm(loader, disable=getattr(loader.dataset.opt, "disable_tqdm", False)):
            dt = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in dt.items()}
            dt = collections.defaultdict(lambda: None, dt)
            dt["video_target"] = [
                {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in item.items()}
                for item in dt["video_target"]
            ]

            output, _ = model(dt, criterion, transformer_input_type, eval_mode=True)
            orig_target_sizes = dt["video_length"][:, 1]
            results = postprocessors["bbox"](output, orig_target_sizes, loader)
            mimic_fine_logits = output["mimic_fine_logits"]
            mimic_overall_logits = output["mimic_overall_logits"]

            for batch_index, video_name in enumerate(dt["video_key"]):
                proposals = []
                boxes = results[batch_index]["boxes"].cpu().numpy()
                raw_boxes = results[batch_index]["raw_boxes"].cpu().numpy()
                aligned_fine = results[batch_index].get("mimic_fine_scores", None)
                for proposal_index in range(len(boxes)):
                    proposal_score = results[batch_index]["scores"][proposal_index].item()
                    if proposal_score < score_threshold:
                        continue
                    fine_score = (
                        aligned_fine[proposal_index].item()
                        if aligned_fine is not None
                        else mimic_fine_logits.sigmoid()[batch_index][results[batch_index]["query_id"][proposal_index]].item()
                    )
                    proposals.append(
                        {
                            "timestamp": boxes[proposal_index].tolist(),
                            "raw_box": raw_boxes[proposal_index].tolist(),
                            "proposal_score": proposal_score,
                            "mimic_fine_score": fine_score,
                            "mimic_overall_score": mimic_overall_logits.sigmoid()[batch_index].item(),
                            "sentence": results[batch_index]["captions"][proposal_index],
                            "sentence_score": results[batch_index]["caption_scores"][proposal_index],
                            "query_id": results[batch_index]["query_id"][proposal_index].item(),
                            "vid_duration": results[batch_index]["vid_duration"].item(),
                            "pred_event_count": results[batch_index]["pred_seq_len"].item(),
                        }
                    )
                predictions[video_name] = proposals
    return predictions


def default_threshold(cli_value, config_value):
    if cli_value is not None:
        return float(cli_value)
    if config_value is not None:
        return float(config_value)
    return 0.5


def choose_events_for_render(predictions, max_events):
    if not predictions:
        return []
    ranked = sorted(predictions, key=lambda item: item["proposal_score"], reverse=True)
    predicted_count = int(max((item.get("pred_event_count", 0) for item in ranked), default=0))
    keep_num = max_events if max_events and max_events > 0 else predicted_count
    if keep_num <= 0:
        keep_num = len(ranked)
    selected = ranked[:keep_num]
    selected = sorted(selected, key=lambda item: (item["timestamp"][0], item["timestamp"][1], -item["proposal_score"]))
    return selected


def classify_label(score, threshold):
    return "CORRECT" if float(score) >= float(threshold) else "ERROR"


def wrap_text_cv2(cv2, text, max_width, font, scale, thickness):
    if not text:
        return [""]
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text_block(cv2, canvas, lines, x, y, font, scale, thickness, color, line_gap):
    current_y = y
    for line in lines:
        cv2.putText(canvas, line, (x, current_y), font, scale, color, thickness, cv2.LINE_AA)
        current_y += line_gap
    return current_y


def active_and_neighbor_events(events, current_time):
    active = [event for event in events if event["timestamp"][0] <= current_time <= event["timestamp"][1]]
    if active:
        active = sorted(active, key=lambda item: item["proposal_score"], reverse=True)
        return active[0], None, None

    past = [event for event in events if event["timestamp"][1] < current_time]
    future = [event for event in events if event["timestamp"][0] > current_time]
    prev_event = max(past, key=lambda item: item["timestamp"][1]) if past else None
    next_event = min(future, key=lambda item: item["timestamp"][0]) if future else None
    return None, prev_event, next_event


def render_captioned_video(
    ego_video_path,
    output_path,
    events,
    sample_info,
    fine_threshold,
    overall_threshold,
    render_fps,
    max_frames,
    panel_ratio,
):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Rendering requires OpenCV. Install it with `pip install opencv-python`."
        ) from exc

    cap = cv2.VideoCapture(str(ego_video_path))
    if not cap.isOpened():
        raise IOError("Failed to open ego video: {}".format(ego_video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if src_fps <= 0:
        src_fps = 25.0
    out_fps = float(render_fps) if render_fps and render_fps > 0 else src_fps

    panel_height = max(int(height * panel_ratio), 120)
    total_height = height + panel_height
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, total_height))
    if not writer.isOpened():
        raise IOError("Failed to open output video writer: {}".format(output_path))

    overall_score = max((event.get("mimic_overall_score", 0.0) for event in events), default=0.0)
    overall_label = classify_label(overall_score, overall_threshold)
    overall_color = (60, 180, 75) if overall_label == "CORRECT" else (50, 70, 220)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = max(width / 1600.0, 0.55)
    body_scale = max(width / 1800.0, 0.50)
    small_scale = max(width / 2200.0, 0.42)
    title_thickness = 2
    body_thickness = 1
    pad_x = 20
    line_gap = int(28 * max(body_scale, 0.6))
    title_gap = int(34 * max(title_scale, 0.7))

    frame_index = 0
    duration = max((event["vid_duration"] for event in events), default=0.0)
    duration = duration if duration > 0 else float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / src_fps

    while True:
        success, frame = cap.read()
        if not success:
            break
        if max_frames and frame_index >= max_frames:
            break

        current_time = frame_index / src_fps
        panel = np.full((panel_height, width, 3), 18, dtype=np.uint8)
        active_event, prev_event, next_event = active_and_neighbor_events(events, current_time)

        cv2.putText(
            panel,
            "{} | {} ({:.2f})".format(sample_info["sample_id"], overall_label, overall_score),
            (pad_x, 34),
            font,
            title_scale,
            overall_color,
            title_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "time {:.2f}s / {:.2f}s".format(current_time, duration),
            (width - 260, 34),
            font,
            body_scale,
            (220, 220, 220),
            body_thickness,
            cv2.LINE_AA,
        )

        timeline_y0 = 52
        timeline_y1 = 72
        cv2.rectangle(panel, (pad_x, timeline_y0), (width - pad_x, timeline_y1), (70, 70, 70), -1)
        for event in events:
            start, end = event["timestamp"]
            x0 = pad_x + int((width - 2 * pad_x) * max(0.0, min(1.0, start / max(duration, 1e-6))))
            x1 = pad_x + int((width - 2 * pad_x) * max(0.0, min(1.0, end / max(duration, 1e-6))))
            label = classify_label(event["mimic_fine_score"], fine_threshold)
            color = (60, 180, 75) if label == "CORRECT" else (50, 70, 220)
            cv2.rectangle(panel, (x0, timeline_y0), (max(x1, x0 + 2), timeline_y1), color, -1)
            if active_event is event:
                cv2.rectangle(panel, (x0, timeline_y0), (max(x1, x0 + 2), timeline_y1), (255, 255, 255), 2)
        cursor_x = pad_x + int((width - 2 * pad_x) * max(0.0, min(1.0, current_time / max(duration, 1e-6))))
        cv2.line(panel, (cursor_x, timeline_y0 - 6), (cursor_x, timeline_y1 + 6), (255, 255, 255), 2)

        text_y = 104
        if active_event is not None:
            step_label = classify_label(active_event["mimic_fine_score"], fine_threshold)
            step_color = (60, 180, 75) if step_label == "CORRECT" else (50, 70, 220)
            header = "Current step | {} | fine {:.2f} | det {:.2f} | [{:.2f}, {:.2f}]s".format(
                step_label,
                active_event["mimic_fine_score"],
                active_event["proposal_score"],
                active_event["timestamp"][0],
                active_event["timestamp"][1],
            )
            cv2.putText(panel, header, (pad_x, text_y), font, body_scale, step_color, body_thickness, cv2.LINE_AA)
        elif next_event is not None:
            header = "Next step | {} | fine {:.2f} | det {:.2f} | starts at {:.2f}s".format(
                classify_label(next_event["mimic_fine_score"], fine_threshold),
                next_event["mimic_fine_score"],
                next_event["proposal_score"],
                next_event["timestamp"][0],
            )
            cv2.putText(panel, header, (pad_x, text_y), font, body_scale, (200, 200, 200), body_thickness, cv2.LINE_AA)
        elif prev_event is not None:
            header = "Last step | {} | fine {:.2f} | det {:.2f} | ended at {:.2f}s".format(
                classify_label(prev_event["mimic_fine_score"], fine_threshold),
                prev_event["mimic_fine_score"],
                prev_event["proposal_score"],
                prev_event["timestamp"][1],
            )
            cv2.putText(panel, header, (pad_x, text_y), font, body_scale, (200, 200, 200), body_thickness, cv2.LINE_AA)
        else:
            cv2.putText(panel, "No predicted steps passed the render threshold.", (pad_x, text_y), font, body_scale, (210, 210, 210), body_thickness, cv2.LINE_AA)

        footer_y = panel_height - 16
        footer = "ego: {}{}".format(
            Path(ego_video_path).name,
            " | exo: {}".format(Path(sample_info["exo_video_path"]).name) if sample_info.get("exo_video_path") else "",
        )
        footer_lines = wrap_text_cv2(cv2, footer, width - 2 * pad_x, font, small_scale, body_thickness)
        draw_text_block(
            cv2,
            panel,
            footer_lines[-1:],
            pad_x,
            footer_y,
            font,
            small_scale,
            body_thickness,
            (150, 150, 150),
            max(18, int(20 * small_scale)),
        )

        combined = np.zeros((total_height, width, 3), dtype=np.uint8)
        combined[:height] = frame
        combined[height:] = panel
        writer.write(combined)
        frame_index += 1

    cap.release()
    writer.release()


def main():
    runtime = load_runtime_modules()
    torch = runtime["torch"]
    opt = parse_opts()
    logger = setup_logging()

    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in opt.gpu_id)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    if getattr(opt, "disable_cudnn", False):
        torch.backends.cudnn.enabled = False

    if opt.eval_device is None:
        opt.eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and opt.eval_device == "cuda":
        logger.warning("CUDA is unavailable; falling back to CPU.")
        opt.eval_device = "cpu"
    opt.device = opt.eval_device
    if not torch.cuda.is_available() and (opt.nthreads is None or opt.nthreads > 0):
        opt.nthreads = 0

    model_path = opt.eval_model_path or getattr(opt, "eval_model_path", None)
    if model_path is None:
        raise ValueError("eval_model_path must be set in YAML or CLI.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("checkpoint not found: {}".format(model_path))

    restore_training_options(opt, os.path.join(os.path.dirname(model_path), "info.json"), logger)
    opt.device = opt.eval_device

    dataset = runtime["PropSeqDataset"](
        [opt.ego_eval_caption_file, opt.exo_eval_caption_file],
        getattr(opt, "visual_feature_folder", None),
        opt.dict_file,
        False,
        getattr(opt, "eval_proposal_type", None),
        opt,
    )
    indices, sample_info_map = select_sample_indices(dataset, opt, logger)
    selected_dataset = DatasetView(dataset, indices)
    loader = create_loader(selected_dataset, opt.batch_size_for_eval, opt.nthreads or 0)

    model, criterion, postprocessors = runtime["build"](opt)
    model.translator = dataset.translator

    checkpoint = torch.load(model_path, map_location=opt.eval_device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.to(opt.eval_device)

    logger.info("Running inference for %d samples...", len(selected_dataset))
    predictions = predict_selected(
        model,
        criterion,
        postprocessors,
        loader,
        device=opt.eval_device,
        transformer_input_type=opt.transformer_input_type,
        score_threshold=opt.proposal_score_threshold,
    )

    fine_threshold = default_threshold(opt.fine_threshold, getattr(opt, "fixed_fine_threshold", None))
    overall_threshold = default_threshold(opt.overall_threshold, getattr(opt, "fixed_overall_threshold", None))

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dump = {
        "meta": {
            "cfg_path": opt.cfg_path,
            "model_path": model_path,
            "fine_threshold": fine_threshold,
            "overall_threshold": overall_threshold,
            "proposal_score_threshold": opt.proposal_score_threshold,
            "max_events": opt.max_events,
        },
        "results": {},
    }

    rendered = 0
    for sample_id, sample_info in sample_info_map.items():
        sample_predictions = predictions.get(sample_info["ego_key"], [])
        events = choose_events_for_render(sample_predictions, opt.max_events)
        prediction_dump["results"][sample_id] = {
            "sample_info": sample_info,
            "events": events,
        }

        ego_video_path = sample_info.get("ego_video_path")
        if not ego_video_path:
            logger.warning("Skip %s because no ego video was resolved.", sample_id)
            continue

        safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", sample_id)
        output_path = output_dir / "{}{}".format(safe_name, opt.output_video_suffix)
        logger.info("Rendering %s -> %s", sample_id, output_path)
        render_captioned_video(
            ego_video_path=ego_video_path,
            output_path=output_path,
            events=events,
            sample_info=sample_info,
            fine_threshold=fine_threshold,
            overall_threshold=overall_threshold,
            render_fps=opt.render_fps,
            max_frames=opt.max_frames,
            panel_ratio=opt.panel_ratio,
        )
        rendered += 1

    json_path = output_dir / "predictions_for_visualization.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(prediction_dump, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    logger.info("Finished. Rendered %d videos. Predictions saved to %s", rendered, json_path)


if __name__ == "__main__":
    main()
