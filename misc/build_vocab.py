# coding:utf-8
import argparse
import json
from pathlib import Path


DEFAULT_INPUT_JSONS = [
    "data/egome/train_Ego_cap.json",
    "data/egome/val_Ego_cap.json",
]
DEFAULT_OUTPUT_PATH = "data/egome/vocabulary_egome.json"
TOKENS_TO_STRIP = [",", ":", "!", "_", ";", "-", ".", "?", "/", '"', "\\n", "\\"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a vocabulary JSON from EgoMe-style caption annotation files."
    )
    parser.add_argument(
        "--input_jsons",
        nargs="+",
        default=DEFAULT_INPUT_JSONS,
        help=(
            "EgoMe-style *_cap.json files. "
            "Example: data/egome/train_Ego_cap.json data/egome/val_Ego_cap.json"
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output vocabulary JSON path. Example: data/egome/vocabulary_egome.json",
    )
    parser.add_argument(
        "--count_threshold",
        type=int,
        default=2,
        help="Minimum token frequency to keep in the vocabulary.",
    )
    return parser.parse_args()


def normalize_sentence(sentence):
    for token in TOKENS_TO_STRIP:
        sentence = sentence.replace(token, " ")
    sentence = " ".join(sentence.strip().lower().split())
    return sentence.split(" ") if sentence else []


def update_word_counts(annotation_path, word_counts):
    with open(annotation_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{annotation_path} must be a dict keyed by video id")

    print(f"video num of {Path(annotation_path).name} {len(data)}")
    for video_id, info in data.items():
        if "sentences" not in info:
            raise KeyError(f"{annotation_path}: {video_id} is missing 'sentences'")
        for sentence in info["sentences"]:
            for word in normalize_sentence(sentence):
                word_counts[word] = word_counts.get(word, 0) + 1


def build_vocab(word_counts, count_threshold):
    word_counts["<bos>"] = int(1e10)
    word_counts["<eos>"] = int(1e10)

    vocab = [word for word, count in word_counts.items() if count >= count_threshold]
    bad_words = [word for word, count in word_counts.items() if count < count_threshold]
    bad_count = sum(word_counts[word] for word in bad_words)

    vocab.append("UNK")
    print("number of vocab:", len(vocab))
    print("number of bad word:", len(bad_words))
    print("number of unks:", bad_count)

    ix_to_word = {i + 1: word for i, word in enumerate(vocab)}
    word_to_ix = {word: i + 1 for i, word in enumerate(vocab)}
    print(len(ix_to_word))
    print(len(word_to_ix))

    return {"ix_to_word": ix_to_word, "word_to_ix": word_to_ix}


def main():
    args = parse_args()
    word_counts = {}

    for input_json in args.input_jsons:
        update_word_counts(input_json, word_counts)

    print("total word:", sum(word_counts.values()))
    vocab_json = build_vocab(word_counts, args.count_threshold)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(vocab_json, handle, ensure_ascii=False)
    print(f"saving vocabulary file to {output_path}")


if __name__ == "__main__":
    main()
