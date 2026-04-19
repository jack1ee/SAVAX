import argparse
import copy
import json
import os

import yaml


UNSET = object()
SUPPORTED_FUSION_TYPES = {"concat_channel", "concat_time", "ego_only", "cross_attn", "cross_deform"}
SUPPORTED_VIEW_EMBED_TYPES = {"none", "token_type", "viewdict"}
SUPPORTED_VISUAL_FEATURE_TYPES = {"tsp", "videomae"}
SUPPORTED_DEVICES = {"cpu", "cuda"}


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_config(cfg_path):
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, "r") as handle:
        current_cfg = yaml.load(handle, Loader=yaml.FullLoader) or {}

    _reject_legacy_keys(current_cfg, cfg_path)

    base_cfg_path = current_cfg.pop("base_cfg_path", None)
    if base_cfg_path is None:
        return _normalize_compat_keys(current_cfg)

    if not os.path.isabs(base_cfg_path):
        base_cfg_path = os.path.join(os.path.dirname(cfg_path), base_cfg_path)

    base_cfg = load_yaml_config(base_cfg_path)
    return _normalize_compat_keys(_deep_merge(base_cfg, current_cfg))


def apply_cli_overrides(config, overrides):
    merged = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is UNSET or value is None:
            continue
        merged[key] = value
    return merged


def make_namespace(config):
    return argparse.Namespace(**_normalize_compat_keys(config))


def load_json_file(path):
    with open(path, "r") as handle:
        return json.load(handle)


def merge_saved_options(opt, saved_opt, excluded_keys=None, logger=None):
    excluded_keys = set(excluded_keys or [])
    current = vars(opt)
    for key, value in saved_opt.items():
        if key in excluded_keys:
            continue
        old_value = current.get(key)
        current[key] = value
        if logger is not None and old_value != value:
            logger.info("Change opt %s : %s --> %s", key, old_value, value)
    return opt


def _reject_legacy_keys(config, source_name):
    invalid_keys = [key for key in config.keys() if "-" in key]
    if invalid_keys:
        raise ValueError(
            "Unsupported legacy config keys in {}: {}. "
            "Use snake_case YAML keys only.".format(source_name, sorted(invalid_keys))
        )


def _normalize_compat_keys(config):
    normalized = copy.deepcopy(config)

    if "hidden_dim" not in normalized and "feature_dim" in normalized:
        normalized["hidden_dim"] = normalized["feature_dim"]

    if "enc_n_points" not in normalized:
        normalized["enc_n_points"] = 4
    if "dec_n_points" not in normalized:
        normalized["dec_n_points"] = 4

    if "visual_feature_folder" not in normalized:
        ego_folder = normalized.get("ego_feature_folder")
        exo_folder = normalized.get("exo_feature_folder")
        if ego_folder and exo_folder:
            normalized["visual_feature_folder"] = [ego_folder, exo_folder]

    normalized.setdefault("criteria_for_best_ckpt", "mimic")
    normalized.setdefault("ec_alpha", 1.0)
    normalized.setdefault("save_all_checkpoint", 0)
    normalized.setdefault("save_checkpoint_every", 1)
    normalized.setdefault("min_epoch_when_save", -1)
    normalized.setdefault("max_es_cnt", 15)

    return normalized


def validate_training_config(config):
    _validate_common_config(config)
    _require_keys(
        config,
        [
            "ego_train_caption_file",
            "exo_train_caption_file",
            "ego_val_caption_file",
            "exo_val_caption_file",
        ],
        "training",
    )


def validate_eval_config(config):
    normalized = copy.deepcopy(config)
    if "visual_feature_type" not in normalized and "eval_visual_feature_type" in normalized:
        normalized["visual_feature_type"] = normalized["eval_visual_feature_type"]
    _validate_common_config(normalized)
    _require_keys(
        normalized,
        [
            "ego_eval_caption_file",
            "exo_eval_caption_file",
            "eval_save_dir",
            "eval_folder",
        ],
        "evaluation",
    )
    eval_device = config.get("eval_device")
    if eval_device is not None and eval_device not in SUPPORTED_DEVICES:
        raise ValueError("Unsupported eval_device: {}.".format(eval_device))


def _validate_common_config(config):
    _require_keys(
        config,
        [
            "id",
            "dict_file",
            "vocab_size",
            "multi_view",
            "ego_feature_folder",
            "exo_feature_folder",
            "visual_feature_type",
            "feature_dim",
            "frame_embedding_num",
            "max_caption_len",
        ],
        "config",
    )

    if config.get("multi_view") is not True:
        raise ValueError("Only EgoMe-style paired multi_view=true is supported.")

    device = config.get("device")
    if device is not None and device not in SUPPORTED_DEVICES:
        raise ValueError("Unsupported device: {}.".format(device))

    fusion_type = config.get("fusion_type")
    if fusion_type is not None and fusion_type not in SUPPORTED_FUSION_TYPES:
        raise ValueError("Unsupported fusion_type: {}.".format(fusion_type))

    view_embed_type = config.get("view_embed_type")
    if view_embed_type is not None and view_embed_type not in SUPPORTED_VIEW_EMBED_TYPES:
        raise ValueError("Unsupported view_embed_type: {}.".format(view_embed_type))

    visual_feature_type = config.get("visual_feature_type")
    if isinstance(visual_feature_type, str):
        visual_feature_type = [visual_feature_type]
    if not isinstance(visual_feature_type, (list, tuple)) or not visual_feature_type:
        raise ValueError("visual_feature_type must be a non-empty string or list of strings.")
    unsupported_vf_types = [vf_type for vf_type in visual_feature_type if vf_type not in SUPPORTED_VISUAL_FEATURE_TYPES]
    if unsupported_vf_types:
        raise ValueError("Unsupported visual_feature_type values: {}.".format(sorted(set(unsupported_vf_types))))


def _require_keys(config, keys, source_name):
    missing = [key for key in keys if key not in config or config[key] in (None, "")]
    if missing:
        raise ValueError("Missing required {} keys: {}.".format(source_name, missing))
