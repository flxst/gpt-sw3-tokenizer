import matplotlib.pyplot as plt
from collections import Counter
import os
from os.path import join, isfile, isdir
import json
from typing import List, Tuple, Dict, Any, Optional
from termcolor import colored
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm
from src.env import Env
from os.path import dirname, abspath
import sys

BASE_DIR = abspath(dirname(dirname(abspath(__file__))))
print(f">>> BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

env = Env("..")
OUTPUT_DIR = env.output


###########################################################################################
# 0. MODELS ###############################################################################
###########################################################################################
def get_models_in_output_dir() -> List[str]:
    _models: List[str] = sorted([
        _model for _model in os.listdir(OUTPUT_DIR)
        if not (_model.endswith("evaluation") or _model.endswith(".txt"))
    ])

    return _models


###########################################################################################
# 1. TOKENIZATION EXAMPLES ################################################################
###########################################################################################
def decode_hack(_decoded_elementwise):
    """
    needs to be improved:
    - should only be applied if add_prefix_space == True & add_whitespace_tokens == 24
    - should only change an element if the next element is a non-whitespace-element
    """
    return [
        elem[:-1]
        if set(elem) == {' '}
        else elem
        for elem in _decoded_elementwise
    ]
    # return "".join(decoded_elementwise_hack)


def display(_example_decoded_per_token, show_linebreak=False, equal_to_original=None, verbose=True):
    newline = "↩\n" if show_linebreak else "↩"
    example_decoded_per_token = [
        elem.replace("\n", newline).replace(" ", "-")
        for elem in _example_decoded_per_token
    ]

    colors = ["red", "blue"]
    for i, elem in enumerate(example_decoded_per_token):
        print(colored(elem, colors[i % len(colors)]), end="")
    print()
    if verbose:
        if equal_to_original is None:
            print(f"> {len(example_decoded_per_token)} tokens")
        else:
            print(f"> equal to original: {equal_to_original}")
    print()


def tokenize_hf(_model: str, _example: str) -> Dict[str, Any]:
    # 1. load tokenizer
    tokenizer_file = join(OUTPUT_DIR, _model, "tokenizer.json")
    tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    # 2. tokenize & de-tokenize
    _texample = dict()
    _texample['encoded'] = tokenizer_fast.encode(_example)
    _texample['tokenized'] = tokenizer_fast.convert_ids_to_tokens(_texample['encoded'])
    _texample['de-tokenized'] = tokenizer_fast.decode(_texample['encoded'])
    _texample['de-tokenized_elementwise'] = [tokenizer_fast.decode(elem) for elem in _texample['encoded']]

    _texample['de-tokenized_elementwise_hack'] = decode_hack(_texample['de-tokenized_elementwise'])

    return _texample


def tokenize_sp(_model: str, _example: str) -> Dict[str, Any]:
    # 1. load tokenizer
    tokenizer_file = join(OUTPUT_DIR, _model, "model.model")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_file)

    # 2. tokenize & de-tokenize
    _texample = dict()
    _texample['encoded'] = sp.encode(_example, out_type=int)
    _texample['tokenized'] = sp.encode(_example, out_type=str)
    _texample['de-tokenized'] = sp.decode(_texample['tokenized'])

    _texample['de-tokenized_elementwise'] = list()
    idx_end = 0
    for i, token in enumerate(_texample['tokenized']):
        if i == 0 and token.startswith("▁"):
            _token = token[1:]
        elif i > 0 and token.startswith("▁"):
            _token = token.replace("▁", " ")
        else:
            _token = token

        if _token.startswith("<") and _token.endswith(">"):
            _token = sp.decode(_token)

        idx_start = _texample['de-tokenized'][idx_end:].find(_token) + idx_end
        idx_end = idx_start + len(_token)
        _texample['de-tokenized_elementwise'].append(_texample['de-tokenized'][idx_start: idx_end])

    return _texample


###########################################################################################
# 2. SUBWORDS #############################################################################
###########################################################################################
def _get_data(model):
    output_dir = join(env.output, model)
    _subword_lengths_file = join(output_dir, "tokenizer_subword_lengths.json")
    assert isfile(_subword_lengths_file), f"ERROR! could not find {_subword_lengths_file}"

    with open(_subword_lengths_file, "r", encoding="utf-8") as f:
        subword_lenghts = json.load(f)
    vocab_size = subword_lenghts.pop("vocab_size")
    mean = subword_lenghts.pop("mean")
    x = [int(elem) for elem in subword_lenghts.keys()]
    y = [int(elem) for elem in subword_lenghts.values()]
    x, y = zip(*sorted(zip(x, y)))
    data = {
        "x": x,
        "y": y,
        "vocab_size": vocab_size,
        "mean": mean,
    }
    return data


def plot_histogram(model1, model2, xlim, ylim):

    data = [_get_data(model1)]
    if model2 is not None:
        data += [_get_data(model2)]

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    for i in range(len(data)):
        x = data[i]["x"]
        y = data[i]["y"]
        # print(f"x = {x}, y = {y}")
        vocab_size = data[i]["vocab_size"]
        mean = data[i]["mean"]

        ax[i].bar(x, y)
        ax[i].set_xlim([0, xlim])
        ax[i].set_ylim([0, ylim])
        ax[i].plot([mean, mean], [0, ylim], color="r", label="mean")
        ax[i].text(1.1*mean, 0.9*ylim, f"mean = {mean:.2f}", color="r")
        ax[i].set_title(f"vocabulary size = {vocab_size}")
        ax[i].set_xlabel("token length")
        ax[i].set_ylabel("occurrences")

    return fig


def compare_vocab(model1,
                  model2,
                  vocab_1: int,
                  vocab_2: int) -> Tuple[Dict[str, int], List[str], List[str]]:

    def get_vocab(_model, _slt: Optional[int] = None) -> List[str]:
        output_dir = join(env.output, _model)
        _vocab_file = join(output_dir, "tokenizer_vocab.json")
        assert isfile(_vocab_file), f"ERROR! could not find {_vocab_file}"

        with open(_vocab_file, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        vocab = list(vocab_dict.keys())

        vocab = [elem.replace("Ġ", "▁") for elem in vocab]  # for compatibility between HF & SP

        if _slt:
            # vocab = [subword for subword in vocab if len(subword) <= _slt]
            vocab = vocab[:_slt]

        return vocab

    # c. get overlap
    vocab1 = get_vocab(model1, vocab_1)
    vocab2 = get_vocab(model2, vocab_2)
    vocab_intersection = list(set(vocab1).intersection(set(vocab2)))
    vocab_only1 = list(set(vocab1) - set(vocab2))
    vocab_only2 = list(set(vocab2) - set(vocab1))

    return {
        "intersection": len(vocab_intersection),
        "only1": len(vocab_only1),
        "only2": len(vocab_only2),
    }, vocab_only1, vocab_only2


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


COLOR = {
    "da": rgb_to_hex(169, 200, 240),
    "sv": rgb_to_hex(205, 188, 250),
    "no": rgb_to_hex(242, 163, 158),
    "en": rgb_to_hex(244, 183, 138),
    "is": rgb_to_hex(161, 227, 167),
    "cd": rgb_to_hex(216, 188, 159),
    "all": "black",
    "all+": "black",
}


def color(_lang):
    return COLOR[_lang.split("-")[0]]


def _get_overview(_model) -> Dict[str, Any]:
    output_dir = join(env.output, _model)
    _overview_file = join(output_dir, "overview.json")
    assert isfile(_overview_file), f"ERROR! could not find {_overview_file}"

    with open(_overview_file, "r", encoding="utf-8") as f:
        overview_dict = json.load(f)

    return overview_dict


def _get_lang_dataset_size_time(_models, verbose=False):
    overview = {_model: _get_overview(_model) for _model in _models}
    if verbose:
        print(overview)

    if len(overview) == 0:
        return None, None, None

    lang = [_model.split("_")[-1][-2:] for _model in _models]
    dataset_size = [float(overview[_model]["data_size_total"][:-1]) for _model in _models]
    time = [float(overview[_model]["time"][:-1]) for _model in _models]
    dataset_size, time, lang = zip(*sorted(zip(dataset_size, time, lang)))

    return lang, dataset_size, time


def plot_overview(_models, verbose=False):
    lang, dataset_size, time = _get_lang_dataset_size_time(_models)
    if verbose:
        print(lang, dataset_size, time)

    if lang is None:
        print("no data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(dataset_size, time, linestyle=None, marker="x")
    ax.set_xlabel("dataset size [GB]")
    ax.set_ylabel("time [s]")
    time_estimate = (time[-1]-time[-2])/(dataset_size[-1]-dataset_size[-2])*1000
    ax.set_title(f"time(1TB) ~ {time_estimate:.0f}s ~ {time_estimate/3600.:.0f}h")
    for i in range(len(dataset_size)):
        ax.plot(dataset_size[i], time[i], marker="d", label=lang[i], color=color(lang[i]))
    ax.legend()


###########################################################################################
# 3. VOCAB SIZE & MULTILINGUALITY #########################################################
###########################################################################################
def _split_models_multilinguality(_models_multilinguality: Dict[str, str], _core: str) -> Dict[str, Any]:
    _ml = dict()
    if len(_models_multilinguality):
        _ml["lang_complete"] = list(_models_multilinguality.keys())
        _ml["lang_all"] = [l for l in _ml["lang_complete"] if l == "all"]
        _ml["lang_pure"] = [l for l in _ml["lang_complete"] if l != "all"]

        _ml["models_complete"] = _models_multilinguality
        _ml["models_all"] = {k: _models_multilinguality[k] for k in _ml["lang_all"]}
        _ml["models_pure"] = {k: _models_multilinguality[k] for k in _ml["lang_pure"]}
    else:
        _ml = {
            k: [] if k.startswith("lang") else {}
            for k in ["lang_complete", "lang_all", "lang_pure", "models_complete", "models_all", "models_pure"]
        }
    return _ml


def _extract_language(_model: str, _core: str) -> str:
    if _model.endswith(_core):
        _language = "all"
    else:
        _language = _model.split(_core)[-1].split("_")[-1]
    return _language


def get_models_multilinguality(_models: List[str], verbose: bool = False) -> Dict[str, Any]:
    if len(_models):
        cores = list()
        for model in _models:
            res = model.split("-v")[-1]
            res = "v" + "_".join(res.split("_", 2)[:2])
            if res in model:
                cores.append(res)
        counts = dict(Counter(cores))
    else:
        counts = {}

    cores = [k for k, v in counts.items() if v > 1]
    assert len(cores) <= 1, f"ERROR! found {len(cores)} cores: {cores} - should be 0 or 1."
    if len(cores) == 1:
        core = cores[0]

        _models_multilinguality_list = [model for model in _models if core in model]
        _models_multilinguality = {_extract_language(model, core): model for model in _models_multilinguality_list}
        if verbose:
            print("counts:", counts)
            print("core:", core)
            print("models_multilinguality", _models_multilinguality)
    else:
        _models_multilinguality = {}
        core = ""

    return _split_models_multilinguality(_models_multilinguality, core)


def get_intersection(_models_multilinguality: Dict[str, str],
                     lang_1: str,
                     lang_2: str,
                     vocab_1: int,
                     vocab_2: int):
    model_1 = _models_multilinguality[lang_1]
    model_2 = _models_multilinguality[lang_2]
    v, _, _ = compare_vocab(model_1, model_2, vocab_1, vocab_2)
    return v["intersection"]


def get_intersections(_ml: Dict[str, Dict],
                      _vocabs_1: List[int],
                      _vocabs_2: List[int]) -> Dict[str, Dict]:
    _models_multilinguality = _ml["models_complete"]
    _timelines = dict()
    if len(_models_multilinguality):
        intersections = {
            lang_1: {
                lang_2: {
                    vocab_1: {
                        vocab_2: get_intersection(_models_multilinguality,
                                                  lang_1,
                                                  lang_2,
                                                  vocab_1,
                                                  vocab_2)
                        for vocab_2 in _vocabs_2
                    }
                    for vocab_1 in _vocabs_1
                }
                for lang_2 in _ml["lang_complete"]
            }
            for lang_1 in _ml["lang_all"]
        }

        _timelines['abs'] = {
            lang_1: {
                vocab_2: {
                    lang_2:
                        [intersections[lang_1][lang_2][vocab_1][vocab_2] for vocab_1 in _vocabs_1]
                    for lang_2 in _ml["lang_pure"]
                }
                for vocab_2 in _vocabs_2
            }
            for lang_1 in _ml["lang_all"]
        }

        _timelines['rel'] = {
            lang_1: {
                vocab_2: {
                    lang_2:
                        [intersections[lang_1][lang_2][vocab_1][vocab_2] / intersections[lang_1][lang_1][vocab_2][
                            vocab_2] for vocab_1 in _vocabs_1]
                    for lang_2 in _ml["lang_pure"]
                }
                for vocab_2 in _vocabs_2
            }
            for lang_1 in _ml["lang_all"]
        }
    else:
        _timelines = None

    return _timelines


def plot_overview_data(_models, verbose=False):
    lang, dataset_size, time = _get_lang_dataset_size_time(_models, verbose)
    if verbose:
        print(lang, dataset_size, time)

    if lang is None:
        print("no data")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].pie(dataset_size,
              labels=lang,
              autopct='%.f%%',
              shadow=False,
              startangle=90,
              colors=[color(lg) for lg in lang])
    ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax[0].set_title('dataset size (fraction)')

    x = lang
    y = dataset_size
    ax[1].bar(x, y, color=[color(lg) for lg in lang])
    ax[1].set_title('dataset size [GB]')


def plot_vocab_size(_model):
    subdir = _model.split("/")[0]

    # 1. get all models that only differ w.r.t. minimum vocabulary
    output_dir = join("..", "output", subdir)
    models = [
        join(subdir, model)
        for model in os.listdir(output_dir)
        if isdir(join(output_dir, model))
    ]

    _model_wo_id = "_".join(_model.split("/")[-1].split("_")[1:])
    _start = _model_wo_id.split("-f")[0]
    _end = _model_wo_id.split("-v")[-1]
    models = [model for model in models if _start in model and model.endswith(_end)]

    model_dict = {
        int(model.split("-f")[-1].split("-")[0]): model
        for model in models
    }
    model_dict = dict(sorted(model_dict.items(), key=lambda item: item[0]))

    # 2. get data from all models
    model_data = {k: _get_data(v) for k, v in model_dict.items()}

    min_frequency = [k for k in model_data.keys()]
    vocab_size = [v["vocab_size"] for v in model_data.values()]
    mean = [v["mean"] for v in model_data.values()]
    # print(min_frequency, vocab_size, mean)

    # 3. plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(min_frequency, vocab_size, color="k", linestyle="-", marker="s")
    ax[0].set_xlabel("minimum_frequency")
    ax[0].set_title("vocabulary size")
    ax[0].set_ylim([0, max(vocab_size)*1.1])
    ax[1].plot(min_frequency, mean, color="k", linestyle="--", marker="s")
    ax[1].set_xlabel("minimum_frequency")
    ax[1].set_title("mean(subword length)")
    ax[1].set_ylim([0, max(mean)*1.1])


def plot_timelines(steps: List[int],
                   steps_2: int,
                   _timelines_all: List[Dict[str, List[float]]],
                   lang: List[str],
                   ylim: List[float],
                   ylabel: List[str],
                   title: List[str]):
    nfigs = len(_timelines_all)
    fig, ax = plt.subplots(1, nfigs, figsize=(8 * nfigs, 5))
    _ax = [ax] if nfigs == 1 else ax

    for nfig in range(nfigs):
        for i, (k, v) in enumerate(_timelines_all[nfig].items()):
            x = steps  # [j for j in range(len(_timelines_all[nfig][k]))]
            y = _timelines_all[nfig][k]
            # print(i)
            # print(x)
            # print(y)
            # print()
            x_full = [elem for elem in x if elem > 50000]
            threshold_empty = len(x) - len(x_full)
            y_full = y[threshold_empty:]
            _ax[nfig].plot(x, y, color=color(k), label=None, marker="s")  # , markerfacecolor="w")
            _ax[nfig].plot(x_full, y_full, color=color(k), label=lang[i], marker="s")
            _ax[nfig].plot(x, [steps_2]*len(x), color="k")
            if ylabel[nfig] == "relative":
                _ax[nfig].plot(x, [1] * len(x), color="k")
                _ax[nfig].text(x[0], 1.03, steps_2)
            else:
                _ax[nfig].plot([0, ylim[nfig]], [0, ylim[nfig]], marker=None, linestyle="--", color="gray")
        _ax[nfig].set_xlabel("multilingual tokenizer vocabulary size")
        _ax[nfig].set_ylabel(ylabel[nfig])
        _ax[nfig].set_title(title[nfig])
        _ax[nfig].set_xlim([0, None])
        _ax[nfig].set_ylim([0, ylim[nfig]])
        _ax[nfig].legend()


###########################################################################################
# 3. Evaluation #2 ########################################################################
###########################################################################################
def get_list_of_results():
    evaluation_dir = join(OUTPUT_DIR, "evaluation")
    results = [
        elem.split("results_")[-1].split(".json")[0]
        for elem in sorted(os.listdir(evaluation_dir))
        if elem.split("/")[-1].startswith("results_")
    ]
    return results


def read_results(_result):
    _results_path = join(OUTPUT_DIR, "evaluation", f"results_{_result}.json")
    with open(_results_path, "r") as file:
        r = json.load(file)
    return r


def retrieve_groups_from_results(_results):
    datasets = list(set([k.split("/")[-1].split("_")[0] for v in _results.values() for k, _ in v.items()]))
    return datasets


def retrieve_parameters_from_results(_group, _results, verbose: bool = False):
    models = list(set(_results.keys()))
    vocabs = sorted(list(set([int(model.split("-v")[1].split("_")[0]) for model in models])))
    vocabs_model = {
        vocab: [
            model
            for model in models
            if f"-v{vocab}_" in model
        ][0]
        for vocab in vocabs
    }
    files = list([elem for elem in _results[models[0]].keys() if f"{_group}_" in elem])

    languages = [file.split("/")[-1].split(".json")[0].split("_")[1] for file in
                 files]  # WORKS ONLY FOR 'wiki_??_t1p'!!!
    languages_files = {k: v for k, v in zip(languages, files)}

    if verbose:
        print("> vocabs:", vocabs)
        print("> vocabs_model:", vocabs_model)
        print("> files:", files)
        print("> languages:", languages)

    return vocabs, vocabs_model, files, languages, languages_files


def extract(_results_filtered, vocabs_models, vocabs, languages_files, languages):
    unk_rate = {
            language: [
                _results_filtered[vocabs_models[vocab]][languages_files[language]]["unk_rate"]
                for vocab in vocabs
            ]
            for language in languages
        }

    try:
        ctcl = {
            language: [
                _results_filtered[vocabs_models[vocab]][languages_files[language]]["ctcl"]
                for vocab in vocabs
            ]
            for language in languages
        }
    except KeyError:  # compatibility with paper v1 data
        ctcl = {
            language: [
                _results_filtered[vocabs_models[vocab]][languages_files[language]]["closeness_to_character_level"]
                for vocab in vocabs
            ]
            for language in languages
        }

    try:
        fertility = {
            language: [
                _results_filtered[vocabs_models[vocab]][languages_files[language]]["fertility"]
                for vocab in vocabs
            ]
            for language in languages
        }
    except KeyError:  # compatibility with paper v1 data
        fertility = {
            language: [
                0
                for _ in vocabs
            ]
            for language in languages
        }

    try:
        proportion = {
            language: [
                _results_filtered[vocabs_models[vocab]][languages_files[language]]["proportion"]
                for vocab in vocabs
            ]
            for language in languages
        }
    except KeyError:  # compatibility with paper v1 data
        proportion = {
            language: [
                0
                for _ in vocabs
            ]
            for language in languages
        }

    return unk_rate, ctcl, fertility, proportion
