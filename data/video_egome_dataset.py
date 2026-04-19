from collections import defaultdict
from itertools import chain
import json
import os
import re

import lmdb
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


def pad_feat_list(feat_list, max_len):
    """Pad list[(T, D)] to (B, T_max, D) and return mask."""
    batch_size = len(feat_list)
    feat_dim = feat_list[0].shape[1]
    tensor = torch.zeros(batch_size, max_len, feat_dim)
    mask = torch.zeros(batch_size, max_len).bool()
    for index, feat in enumerate(feat_list):
        length = feat.shape[0]
        if not feat.flags.writeable:
            feat = feat.copy()
        tensor[index, :length] = torch.from_numpy(feat)
        mask[index, :length] = True
    return tensor, mask


def worker_init_fn(worker_id):
    dataset = torch.utils.data.get_worker_info().dataset
    dataset._lazy_open_env()


class Translator(object):
    def __init__(self, translator_json, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = json.load(open(translator_json, "r"))
        assert self.vocab_size == len(self.vocab["word_to_ix"].keys())
        self.vocab["word_to_ix"] = defaultdict(lambda: self.vocab_size, self.vocab["word_to_ix"])
        self.vocab["ix_to_word"] = defaultdict(lambda: self.vocab_size, self.vocab["ix_to_word"])
        print("load translator, total_vocab: {}".format(len(self.vocab["ix_to_word"])))

    def translate(self, sentence, max_len):
        tokens = [",", ":", "!", "_", ";", "-", ".", "?", "/", '"', "\\n", "\\", "."]
        for token in tokens:
            sentence = sentence.replace(token, " ")
        sentence_split = sentence.replace(".", " . ").replace(",", " , ").lower().split()
        return np.array([0] + [self.vocab["word_to_ix"][word] for word in sentence_split][: max_len - 2] + [0])

    def rtranslate(self, sent_ids):
        for index in range(len(sent_ids)):
            if sent_ids[index] == 0:
                sent_ids = sent_ids[:index]
                break
        if len(sent_ids):
            return " ".join([self.vocab["ix_to_word"][str(idx)] for idx in sent_ids]) + "."
        return ""


class PropSeqDataset(Dataset):
    """
    Unified EgoMe-style paired ego/exo dataset.

    This dataset only supports paired EgoMe-style annotations. Other datasets must
    be converted into the same JSON and feature layout before training/evaluation.
    """

    def __init__(self, anno_file, feature_folder, translator_pickle, is_training, proposal_type, opt, vf_types=None):
        super(PropSeqDataset, self).__init__()
        if not getattr(opt, "multi_view", True):
            raise ValueError("Only paired EgoMe-style multi_view data is supported.")
        if not isinstance(anno_file, (list, tuple)) or len(anno_file) != 2:
            raise ValueError("anno_file must be [ego_annotation_json, exo_annotation_json].")

        self.opt = opt
        self.is_training = is_training
        self.proposal_type = proposal_type
        self.max_caption_len = opt.max_caption_len
        self.gt_proposal_sample_num = opt.gt_proposal_sample_num
        self.feature_sample_rate = opt.feature_sample_rate
        self.frame_embedding_num = opt.frame_embedding_num
        self.data_rescale = bool(opt.data_rescale)
        self.data_rescale_type = opt.data_rescale_type
        self.data_norm = opt.data_norm
        self.feature_dim = opt.feature_dim
        self.translator = Translator(translator_pickle, opt.vocab_size)
        self.visual_feature_types = vf_types or opt.visual_feature_type
        if isinstance(self.visual_feature_types, str):
            self.visual_feature_types = [self.visual_feature_types]

        self.ego_folder = opt.ego_feature_folder
        self.exo_folder = opt.exo_feature_folder
        if not self.ego_folder or not self.exo_folder:
            raise ValueError("Both ego_feature_folder and exo_feature_folder must be configured.")
        self.anno_ego = self._load_annotation_file(anno_file[0])
        self.anno_exo = self._load_annotation_file(anno_file[1])

        self._env_ego = None
        self._env_exo = None
        self._shape_ego = None
        self._shape_exo = None

        ego_map = {self._strip_suffix(key): key for key in self.anno_ego.keys()}
        exo_map = {self._strip_suffix(key): key for key in self.anno_exo.keys()}
        common_ids = sorted(set(ego_map.keys()) & set(exo_map.keys()))
        if not common_ids:
            raise RuntimeError("Ego and Exo annotations have no paired samples.")

        missing_ego = sorted(set(exo_map.keys()) - set(ego_map.keys()))
        missing_exo = sorted(set(ego_map.keys()) - set(exo_map.keys()))
        if missing_ego or missing_exo:
            print("[Warning] unpaired samples detected; only paired intersection will be used.")
            if missing_ego:
                print("  - Missing ego pairs: {} (e.g. {})".format(len(missing_ego), missing_ego[:3]))
            if missing_exo:
                print("  - Missing exo pairs: {} (e.g. {})".format(len(missing_exo), missing_exo[:3]))

        self.keys = common_ids
        self.id2ego = {sample_id: ego_map[sample_id] for sample_id in common_ids}
        self.id2exo = {sample_id: exo_map[sample_id] for sample_id in common_ids}

        print("loaded EgoMe-style paired dataset: {} samples".format(len(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        for env in (self._env_ego, self._env_exo):
            if env is not None:
                env.close()

    def _load_annotation_file(self, path):
        if not os.path.exists(path):
            raise IOError("annotation file not found: {}".format(path))
        annotation = json.load(open(path, "r"))
        invalid_ids = self._load_invalid_ids()
        if not invalid_ids:
            return annotation
        filtered = {}
        for key, value in annotation.items():
            base_id = self._strip_suffix(key)
            if key in invalid_ids or base_id in invalid_ids or key[:13] in invalid_ids:
                continue
            filtered[key] = value
        return filtered

    def _load_invalid_ids(self):
        invalid_ids = set()
        for json_path in getattr(self.opt, "invalid_video_json", []):
            invalid_values = json.load(open(json_path, "r"))
            if isinstance(invalid_values, dict):
                invalid_ids.update(invalid_values.keys())
            else:
                invalid_ids.update(invalid_values)
        return invalid_ids

    def _lazy_open_env(self):
        if self._env_ego is None:
            self._env_ego, self._shape_ego = self._open_lmdb(self.ego_folder)
            self._env_exo, self._shape_exo = self._open_lmdb(self.exo_folder)

    def _open_lmdb(self, folder):
        lmdb_path = os.path.join(folder, "features.lmdb")
        shape_file = os.path.join(folder, "shapes.json")
        if not os.path.exists(lmdb_path):
            return None, None
        if not os.path.exists(shape_file):
            raise IOError("LMDB feature folder {} is missing shapes.json".format(folder))
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
        shapes = json.load(open(shape_file, "r"))
        return env, shapes

    def _strip_suffix(self, key):
        for suffix in ("_ego", "_exo"):
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    def _load_feature_from_lmdb(self, video_key, folder):
        self._lazy_open_env()
        env = self._env_ego if folder == self.ego_folder else self._env_exo
        shapes = self._shape_ego if folder == self.ego_folder else self._shape_exo
        if env is None or video_key not in shapes:
            return None
        with env.begin(buffers=True) as txn:
            buffer = txn.get(video_key.encode())
        if buffer is None:
            return None
        feat = np.frombuffer(buffer, dtype=np.float32).reshape(shapes[video_key])
        return self._postprocess_feature_array(feat)

    def _load_feature_from_files(self, video_key, folder):
        feat_list = []
        for vf_type in self.visual_feature_types:
            feat, _ = get_feats(video_key, vf_type, folder, data_norm=self.data_norm)
            feat_list.append(self._postprocess_feature_array(feat))
        if len(feat_list) == 1:
            return feat_list[0]
        return np.concatenate(feat_list, axis=-1)

    def _postprocess_feature_array(self, feat):
        feat = feat[:: self.feature_sample_rate]
        if self.data_rescale:
            feat = resize_feature(feat, self.frame_embedding_num, self.data_rescale_type)
        return feat

    def load_single_view(self, video_key, folder):
        feat = self._load_feature_from_lmdb(video_key, folder)
        if feat is None:
            feat = self._load_feature_from_files(video_key, folder)
        if feat.ndim != 2:
            raise ValueError("feature for {} must be 2D, got {}".format(video_key, feat.shape))
        if feat.shape[1] != self.feature_dim:
            raise AssertionError("{}: feature dim {} != {}".format(video_key, feat.shape[1], self.feature_dim))
        return feat

    def process_time_step(self, duration, timestamps_list, feature_length):
        duration = np.array(duration)
        timestamps = np.array(timestamps_list)
        feature_length = np.array(feature_length)
        featstamps = feature_length * timestamps / duration
        featstamps = np.minimum(featstamps, feature_length - 1).astype("int")
        featstamps = np.maximum(featstamps, 0).astype("int")
        return featstamps.tolist()

    def _validate_annotation(self, video_key, annotation):
        required_keys = ["duration", "timestamps", "sentences"]
        missing = [key for key in required_keys if key not in annotation]
        if missing:
            raise KeyError("{} missing annotation keys: {}".format(video_key, missing))
        if len(annotation["timestamps"]) != len(annotation["sentences"]):
            raise ValueError("{} has mismatched timestamps/sentences lengths".format(video_key))
        if not annotation["timestamps"]:
            raise ValueError("{} has no timestamps/sentences; empty samples are not supported.".format(video_key))
        if annotation["duration"] <= 0:
            raise ValueError("{} has non-positive duration {}".format(video_key, annotation["duration"]))
        for timestamp in annotation["timestamps"]:
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                raise ValueError("{} has invalid timestamp entry {}".format(video_key, timestamp))
            if timestamp[0] > timestamp[1]:
                raise ValueError("{} has reversed timestamp {}".format(video_key, timestamp))

    def _sample_indices(self, item_count):
        sample_num = min(item_count, self.gt_proposal_sample_num)
        if sample_num == 0:
            return np.array([], dtype=np.int64)
        if self.is_training:
            return np.random.choice(item_count, sample_num, replace=False)
        return np.arange(sample_num)

    def _parse_caption_prefix(self, ego_key, sentence):
        match = re.match(r"^([A-Z]):\s*(.*)$", sentence)
        if not match:
            return None, sentence

        prefix, clean_sentence = match.group(1), match.group(2)
        if prefix == "F":
            return 0, clean_sentence
        if prefix == "T":
            return 1, clean_sentence

        raise ValueError(
            "{} uses unsupported label prefix in caption '{}'. "
            "Use 'F:' for erroneous steps, 'T:' for correct steps, or no prefix.".format(ego_key, sentence)
        )

    def _normalize_captions_and_labels(self, ego_key, captions):
        mimic_fine = []
        cleaned_captions = []
        is_false_video = "FALSE" in ego_key.upper()

        for sentence in captions:
            label_from_prefix, cleaned_sentence = self._parse_caption_prefix(ego_key, sentence)
            cleaned_captions.append(cleaned_sentence)
            if label_from_prefix is None:
                mimic_fine.append(1)
            else:
                mimic_fine.append(label_from_prefix)

        if not is_false_video and any(label == 0 for label in mimic_fine):
            raise ValueError(
                "{} is not marked as FALSE but contains 'F:' step labels.".format(ego_key)
            )

        mimic_overall = [0] if is_false_video else [1]
        return cleaned_captions, mimic_fine, mimic_overall

    def __getitem__(self, idx):
        sample_id = self.keys[idx]
        ego_key = self.id2ego[sample_id]
        exo_key = self.id2exo[sample_id]

        ego_ann = self.anno_ego[ego_key]
        exo_ann = self.anno_exo[exo_key]
        self._validate_annotation(ego_key, ego_ann)
        self._validate_annotation(exo_key, exo_ann)

        ego_feat = self.load_single_view(ego_key, self.ego_folder)
        exo_feat = self.load_single_view(exo_key, self.exo_folder)

        captions = ego_ann["sentences"]
        gt_timestamps = ego_ann["timestamps"]
        action_labels = ego_ann.get("action_labels", [0] * len(gt_timestamps))
        if max(action_labels or [0]) > self.opt.num_classes:
            raise AssertionError("{}: invalid action label id {}".format(ego_key, max(action_labels)))

        keep_ids = self._sample_indices(len(gt_timestamps))
        captions = [captions[index] for index in keep_ids]
        gt_timestamps = [gt_timestamps[index] for index in keep_ids]
        action_labels = [action_labels[index] for index in keep_ids]

        captions, mimic_fine_label, mimic_overall_label = self._normalize_captions_and_labels(ego_key, captions)
        caption_label = [np.array(self.translator.translate(sentence, self.max_caption_len)) for sentence in captions]
        gt_featstamps = self.process_time_step(ego_ann["duration"], gt_timestamps, ego_feat.shape[0])

        return {
            "feats_ego": ego_feat,
            "feats_exo": exo_feat,
            "gt_featstamps": gt_featstamps,
            "gt_timestamps": gt_timestamps,
            "action_labels": action_labels,
            "mimic_fine_label": mimic_fine_label,
            "mimic_overall_label": mimic_overall_label,
            "caption_label": caption_label,
            "captions": captions,
            "duration": ego_ann["duration"],
            "duration_exo": exo_ann["duration"],
            "key": ego_key,
        }

    def collate_fn(self, batch):
        batch_size = len(batch)
        max_video_length = self.frame_embedding_num

        ego_feats = [item["feats_ego"] for item in batch]
        exo_feats = [item["feats_exo"] for item in batch]
        video_tensor_exo, video_mask_exo = pad_feat_list(exo_feats, max_video_length)

        feature_size = ego_feats[0].shape[1]
        gt_timestamps_list = [item["gt_featstamps"] for item in batch]
        labels = [item["action_labels"] for item in batch]
        mimic_overall_label = [item["mimic_overall_label"] for item in batch]
        mimic_fine_label = [item["mimic_fine_label"] for item in batch]
        caption_list = [item["caption_label"] for item in batch]
        gt_raw_timestamps = [item["gt_timestamps"] for item in batch]
        raw_duration = [item["duration"] for item in batch]
        raw_caption = [item["captions"] for item in batch]
        key = [item["key"] for item in batch]

        max_caption_length = max(chain(*[[len(caption) for caption in captions] for captions in caption_list]))
        total_caption_num = sum(chain([len(captions) for captions in caption_list]))
        max_caption_num = max(len(captions) for captions in caption_list)

        video_tensor = torch.zeros(batch_size, max_video_length, feature_size).float()
        video_length = torch.zeros(batch_size, 3).float()
        video_mask = torch.zeros(batch_size, max_video_length).bool()

        caption_tensor = torch.zeros(total_caption_num, max_caption_length).long()
        caption_length = torch.zeros(total_caption_num).long()
        caption_mask = torch.zeros(total_caption_num, max_caption_length).bool()
        caption_gather_idx = torch.zeros(total_caption_num).long()
        gt_boxes_tensor = torch.zeros(batch_size, max_caption_num, 2)

        total_caption_idx = 0
        for batch_index in range(batch_size):
            video_len = ego_feats[batch_index].shape[0]
            gt_proposal_length = len(gt_timestamps_list[batch_index])

            video_tensor[batch_index, :video_len] = torch.from_numpy(ego_feats[batch_index].copy())
            video_length[batch_index, 0] = float(video_len)
            video_length[batch_index, 1] = raw_duration[batch_index]
            video_length[batch_index, 2] = gt_proposal_length
            video_mask[batch_index, :video_len] = True

            caption_gather_idx[total_caption_idx : total_caption_idx + gt_proposal_length] = batch_index
            gt_boxes_tensor[batch_index, :gt_proposal_length] = torch.tensor(
                [
                    [
                        (timestamp[1] + timestamp[0]) / (2 * raw_duration[batch_index]),
                        (timestamp[1] - timestamp[0]) / raw_duration[batch_index],
                    ]
                    for timestamp in gt_raw_timestamps[batch_index]
                ]
            ).float()

            for caption_offset, captioning in enumerate(caption_list[batch_index]):
                caption_len = len(captioning)
                caption_length[total_caption_idx + caption_offset] = caption_len
                caption_tensor[total_caption_idx + caption_offset, :caption_len] = torch.from_numpy(captioning)
                caption_mask[total_caption_idx + caption_offset, :caption_len] = True
            total_caption_idx += gt_proposal_length

        gt_boxes_mask = (gt_boxes_tensor != 0).sum(2) > 0
        targets = []
        for batch_index, video_id in enumerate(key):
            targets.append(
                {
                    "boxes": torch.tensor(
                        [
                            [
                                (timestamp[1] + timestamp[0]) / (2 * raw_duration[batch_index]),
                                (timestamp[1] - timestamp[0]) / raw_duration[batch_index],
                            ]
                            for timestamp in gt_raw_timestamps[batch_index]
                        ]
                    ).float(),
                    "labels": torch.tensor(labels[batch_index]).long(),
                    "mimic_overall_label": torch.tensor(mimic_overall_label[batch_index]).long(),
                    "mimic_fine_label": torch.tensor(mimic_fine_label[batch_index]).long(),
                    "masks": None,
                    "image_id": video_id,
                }
            )

        dt = {
            "video": {
                "tensor": video_tensor,
                "length": video_length,
                "mask": video_mask,
                "key": list(key),
                "target": targets,
            },
            "gt": {
                "featstamps": gt_timestamps_list,
                "timestamp": list(gt_raw_timestamps),
                "gather_idx": caption_gather_idx,
                "boxes": gt_boxes_tensor,
                "boxes_mask": gt_boxes_mask,
                "view_labels": None,
            },
            "cap": {
                "tensor": caption_tensor,
                "length": caption_length,
                "mask": caption_mask,
                "raw": list(raw_caption),
            },
            "video_tensor_exo": video_tensor_exo,
            "video_mask_exo": video_mask_exo,
            "duration_exo": torch.tensor([item["duration_exo"] for item in batch]).float(),
        }

        flat_dt = {}
        for key1, value1 in dt.items():
            if isinstance(value1, dict):
                for key2, value2 in value1.items():
                    flat_dt["{}_{}".format(key1, key2)] = value2
            else:
                flat_dt[key1] = value1
        return flat_dt


def resize_feature(input_data, new_size, sample_method):
    fixed_len = new_size
    time_length = input_data.shape[0]
    if sample_method == "nearest":
        original_size = len(input_data)
        if original_size == 1:
            input_data = np.reshape(input_data, [-1])
            return np.stack([input_data] * new_size)
        x = np.array(range(original_size))
        interpolator = interp1d(x, input_data, axis=0, kind=sample_method)
        x_new = [index * float(original_size - 1) / (new_size - 1) for index in range(new_size)]
        return interpolator(x_new)
    if sample_method == "longer_uniform":
        if time_length <= fixed_len:
            return input_data
        step = float(time_length) / float(fixed_len)
        indices = (np.arange(fixed_len) * step).astype(int)
        return input_data[indices]
    raise ValueError("unsupported data_rescale_type: {}".format(sample_method))


def read_file(path, feat_dim, mean=0.0, var=1.0, data_norm=False):
    if not os.path.exists(path):
        raise IOError("feature file not found: {}".format(path))

    extension = path.split(".")[-1]
    if extension != "npy":
        raise NotImplementedError("Only .npy feature files are supported, got: {}".format(path))
    feats = np.load(path)

    if data_norm:
        feats = (feats - mean) / np.sqrt(var)
    if feats.ndim == 1:
        assert feats.shape[0] == feat_dim, "{}: {}".format(path, feats.shape)
        feats = feats[None, :]
    assert feats.shape[1] == feat_dim, "{}: {}".format(path, feats.shape)
    return feats, False


def get_feats(key, vf_type, vf_folder, data_norm=False):
    mean = 0
    var = 0
    if vf_type == "tsp":
        feat_dim = 512
        path = os.path.join(vf_folder, key + ".npy")
    elif vf_type == "videomae":
        feat_dim = 768
        path = os.path.join(vf_folder, key + ".npy")
    else:
        raise ValueError("Unsupported visual feature type '{}'. Supported types: ['tsp', 'videomae'].".format(vf_type))

    feats, padding = read_file(path, feat_dim, mean, var, data_norm)
    if len(feats.shape) == 1:
        assert feats.shape[0] == feat_dim, "load {} error, got shape {}".format(path, feats.shape)
    assert feats.shape[1] == feat_dim, "load {} error, got shape {}".format(path, feats.shape)
    return feats, padding
