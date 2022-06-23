import argparse
from typing import List, Dict
from os.path import isfile
import json


def analyze_data(_data_files: List[str]) -> Dict[str, Dict[str, int]]:
    """

    Args:
        _data_files: e.g. ["data/file-is.json", "data/file-sv.json", ..]

    Returns:
        _stats: e.g. {
            'char': {
                'is': 53560978,
                'da': 381259111,
                'no': 644423354,
                'sv': 1372789805,
                'en': 14945086023,
                'other': 4560544,
            },
            'utf8bytes': {..},
            'words': {..},
        }
    """
    lang = ["is", "da", "no", "sv", "en"]
    _stats = {
        "char": {lg: 0 for lg in lang},
        "utf8bytes": {lg: 0 for lg in lang},
        "words": {lg: 0 for lg in lang},
    }

    for k in _stats.keys():
        _stats[k]["other"] = 0

    for _data_file in _data_files:
        with open(_data_file, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            if result["keep"] == 1:
                _lang = result["lang"]
                if _lang not in lang:
                    _lang = "other"
                _stats["char"][_lang] += result["len_char"]
                _stats["utf8bytes"][_lang] += result["len_utf8bytes"]
                _stats["words"][_lang] += result["len_words"]

    return _stats


def compute_upsampling_need(_stats: Dict[str, Dict[str, int]], alpha: float) -> Dict[str, Dict[str, int]]:
    """

    Args:
        _stats: e.g. {
            'char': {
                'is': 53560978,
                'da': 381259111,
                'no': 644423354,
                'sv': 1372789805,
                'en': 14945086023,
                'other': 4560544,
            },
            'utf8bytes': {..},
            'words': {..},
        }
        alpha: e.g. 0.8

    Returns:
        _upsampling_need: e.g. {
            'char': {
                'is': 53560978,
                'da': 381259111,
                'no': 644423354,
                'sv': 1372789805,
                'en': 14945086023,
                'other': 4560544,
            },
            'utf8bytes': {..},
            'words': {..},
        }
    """
    _upsampling_need = {k1: {k2: 0 for k2 in _stats[k1].keys()} for k1 in _stats.keys()}

    def compute_upscaled_f(_p_dict: Dict[str, float], _alpha: float) -> Dict[str, float]:

        # sort such that highest value is at last position
        _p_dict = {k: v for k, v in sorted(_p_dict.items(), key=lambda x: x[1])}

        _keys = list(_p_dict.keys())
        _p = list(_p_dict.values())

        # like in notebook - start
        denominator = sum([elem ** _alpha for elem in _p])
        pure_q = [elem ** _alpha / denominator for elem in _p]
        # pure_f = [q_elem / p_elem for q_elem, p_elem in zip(pure_q, _p)]
        factor = _p[-1] / pure_q[-1]
        upscaled_q = [elem * factor for elem in pure_q]
        upscaled_f = [q_elem / p_elem for q_elem, p_elem in zip(upscaled_q, _p)]
        # like in notebook - end

        upscaled_f_dict = {k: v for k, v in zip(_keys, upscaled_f)}
        upscaled_f_dict["other"] = 1.0

        return upscaled_f_dict

    print()
    print(f"> compute upsampling factors (alpha={alpha}):")
    _upsampling_f = {k: compute_upscaled_f(_stats[k], alpha) for k in _stats.keys()}
    for k, v in _upsampling_f.items():
        print(k, v)

    _upsampling_need = {
        k1: {
            k2: int((v2-1.0) * _stats[k1][k2])
            for k2, v2 in v1.items()
        }
        for k1, v1 in _upsampling_f.items()
    }

    return _upsampling_need


def upsample_data(_lang: str,
                  _alpha: float,
                  _data_files: List[str],
                  _upsampling_need: int,
                  _stats: int,
                  _total_file: str):
    """
    writes upsampled data for lang to _total_file with special lang suffix

    Args:
        _lang: e.g. "is"
        _alpha: e.g. 0.8
        _data_files: e.g. ["data/file-is.json", "data/file-sv.json", ..]
        _upsampling_need: e.g. 111627017
        _stats: e.g. 53560978
        _total_file: e.g.
    """
    _total_file_lang = _total_file.replace(".jsonl", f"_upsampled_a{_alpha}_{_lang}.jsonl")

    _data_lang = list()
    for _data_file in _data_files:
        with open(_data_file, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            if result["keep"] == 1:
                print(result)
                _lang_found = result["lang"]
                if _lang_found == _lang:
                    _data_lang.append(result)
                break

    _upsampled_data = _data_lang[0]  # TODO
    with open(_total_file_lang, "w", encoding="utf-8") as f:
        for _line in _upsampled_data:
            f.write(json.dumps(_line))

    print(f"> wrote upsampled data file {_total_file_lang}")


def upsampling(_data_files: List[str], _stats_file: str, _total_file: str, alpha: float):
    """
    Upsamples data in _data_files using alpha parameter

    Args:
        _data_files: e.g. ["data/file-is.json", "data/file-sv.json", ..]
        _stats_file: e.g. "data/file-stats.json"
        _total_file: e.g. "data/file-total.json"
        alpha: e.g. 0.8

    Returns:
        _upsampled_data_file: e.g. "data/file-upsampled.json"
    """
    # 1. analyze _data_files
    print()
    if not isfile(_stats_file):
        print(f"> stats file {_stats_file} does not exist")
        # compute stats
        print(_data_files)
        stats = analyze_data(_data_files)

        # save stats
        with open(_stats_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(stats))
        print(f"> wrote stats file {_stats_file}")
    else:
        # load stats
        with open(_stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"> read stats file {_stats_file}")

    for k, v in stats.items():
        print(k, v)

    upsampling_need = compute_upsampling_need(stats, alpha)

    print()
    print(f"> compute upsampling need (alpha={alpha}):")
    for k, v in upsampling_need.items():
        print(k, v)


def main(args):
    upsampling(args.dataset_files, args.stats, args.total, args.alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_files", nargs='+', type=str, default=["data/test.json"])
    parser.add_argument("--stats", type=str, default="data/stats.json")
    parser.add_argument("--total", type=str, default="data/total.json")
    parser.add_argument("--alpha", type=float, default=1.0)
    _args = parser.parse_args()

    main(_args)
