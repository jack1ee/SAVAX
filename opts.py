import argparse
import os
import time

from misc.config_utils import (
    UNSET,
    apply_cli_overrides,
    load_yaml_config,
    make_namespace,
    validate_training_config,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = v.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True, help="config file")
    parser.add_argument("--id", type=str, default=UNSET, help="run id")
    parser.add_argument("--gpu_id", type=str, nargs="+", default=UNSET)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=UNSET)
    parser.add_argument("--disable_tqdm", action="store_true", default=UNSET)
    parser.add_argument("--seed", type=int, default=UNSET)
    parser.add_argument("--random_seed", action="store_true", default=UNSET)
    parser.add_argument("--disable_cudnn", type=str2bool, default=UNSET)
    parser.add_argument("--debug", action="store_true", default=UNSET)
    parser.add_argument("--start_from", type=str, default=UNSET, help="run id to resume from")
    parser.add_argument("--start_from_mode", type=str, choices=["best", "last"], default=UNSET)
    parser.add_argument("--pretrain", type=str, choices=["full", "encoder", "decoder", "none"], default=UNSET)
    parser.add_argument("--pretrain_path", type=str, default=UNSET)
    return parser


def parse_opts():
    parser = build_train_parser()
    cli_args = parser.parse_args()

    config = load_yaml_config(cli_args.cfg_path)
    overrides = {key: value for key, value in vars(cli_args).items() if key != "cfg_path"}
    config = apply_cli_overrides(config, overrides)
    config["cfg_path"] = cli_args.cfg_path
    validate_training_config(config)
    args = make_namespace(config)

    if getattr(args, "random_seed", False):
        import random

        seed = int(random.random() * 1000)
        new_id = args.id + "_seed{}".format(seed)
        save_folder = os.path.join(args.save_dir, new_id)
        while os.path.exists(save_folder):
            seed = int(random.random() * 1000)
            new_id = args.id + "_seed{}".format(seed)
            save_folder = os.path.join(args.save_dir, new_id)
        args.id = new_id
        args.seed = seed

    if getattr(args, "debug", False):
        args.id = "debug_" + time.strftime("%y%m%d_%H%M%S", time.localtime())
        args.save_checkpoint_every = 1

    if args.caption_decoder_type == "none":
        assert args.caption_loss_coef == 0
        assert args.set_cost_caption == 0

    print("args.id: {}".format(args.id))
    return args


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
