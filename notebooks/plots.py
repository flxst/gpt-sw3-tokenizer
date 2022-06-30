import matplotlib.pyplot as plt
import os
from os.path import join, isfile, isdir
import json
from typing import List, Tuple, Dict, Any, Optional


def _get_data(model):
    output_dir = join("..", "output", model)
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
    print(len(data))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    for i in range(len(data)):
        x = data[i]["x"]
        y = data[i]["y"]
        vocab_size = data[i]["vocab_size"]
        mean = data[i]["mean"]

        ax[i].bar(x, y)
        ax[i].set_xlim([0, xlim])
        ax[i].set_ylim([0, ylim])
        ax[i].plot([mean, mean], [0, ylim], color="r", label="mean")
        ax[i].set_title(f"vocab_size = {vocab_size} | mean = {mean:.2f}")
        ax[i].set_xlabel("subword length")
        ax[i].set_ylabel("occurrences")

    return fig


def compare_vocab(model1,
                  model2,
                  vocab_1: int,
                  vocab_2: int) -> Tuple[Dict[str, int], List[str], List[str]]:

    def get_vocab(_model, _slt: Optional[int] = None) -> List[str]:
        output_dir = join("..", "output", _model)
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


COLOR = {
    "da": "red",
    "sv": "orange",
    "no": "purple",
    "en": "gray",
    "is": "blue",
    "all": "black",
    "all+": "black",
}


def color(_lang):
    return COLOR[_lang.split("-")[0]]


def _get_overview(_model) -> Dict[str, Any]:
    output_dir = join("..", "output", _model)
    _overview_file = join(output_dir, "overview.json")
    assert isfile(_overview_file), f"ERROR! could not find {_overview_file}"

    with open(_overview_file, "r", encoding="utf-8") as f:
        overview_dict = json.load(f)

    return overview_dict


def _get_lang_dataset_size_time(_models):
    overview = {_model: _get_overview(_model) for _model in _models}
    print(overview)

    if len(overview) == 0:
        return None, None, None

    lang = [_model.split("_3")[-1] for _model in _models]
    dataset_size = [float(overview[_model]["data_size_total"][:-1]) for _model in _models]
    time = [float(overview[_model]["time"][:-1]) for _model in _models]
    dataset_size, time, lang = zip(*sorted(zip(dataset_size, time, lang)))

    return lang, dataset_size, time


def plot_overview(_models):

    lang, dataset_size, time = _get_lang_dataset_size_time(_models)
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


def plot_overview_data(_models):
    lang, dataset_size, time = _get_lang_dataset_size_time(_models)
    print(lang, dataset_size, time)

    if lang is None:
        print("no data")
        return

    # lang = lang[:-1]
    # dataset_size = dataset_size[:-1]

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    # labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    # sizes = [15, 30, 45, 10, 10]
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
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
    ax[1].set_title('dataset size')

    x = lang
    y = [250000]*len(lang)  # [56654, 99452, 150540, 138835, 148386]  # TODO
    print(x, y, [color(lg) for lg in lang])
    ax[2].bar(x, y, color=[color(lg) for lg in lang])
    ax[2].set_title('vocab size')


def plot_timelines(steps: List[int],
                   steps_2: int,
                   _timelines_all: List[Dict[str, List[float]]],
                   lang: List[str],
                   ylim: List[float],
                   ylabel: List[str],
                   title: List[str]):
    nfigs = len(_timelines_all)
    fig, ax = plt.subplots(1, max(2, nfigs), figsize=(8*max(2, nfigs), 5))
    for nfig in range(nfigs):
        for i, (k, v) in enumerate(_timelines_all[nfig].items()):
            x = steps  # [j for j in range(len(_timelines_all[nfig][k]))]
            y = _timelines_all[nfig][k]
            # print(i)
            # print(x)
            # print(y)
            # print()
            ax[nfig].plot(x, y, color=color(k), label=lang[i], marker="s")
            ax[nfig].plot(x, [steps_2]*len(x), color="k")
            if nfig == nfigs - 1:
                ax[nfig].plot(x, [1] * len(x), color="k")
        ax[nfig].set_xlabel("common tokenizer vocab size")
        ax[nfig].set_ylabel(ylabel[nfig])
        ax[nfig].set_title(title[nfig])
        ax[nfig].set_ylim([0, ylim[nfig]])
        ax[nfig].legend()


def plot_vocab_size(_model):
    # print(_model)

    # 1. get all models that only differ w.r.t. minimum vocabulary
    output_dir = join("..", "output")
    models = [subdir for subdir in os.listdir(output_dir) if isdir(join(output_dir, subdir))]
    _model_wo_id = "_".join(_model.split("_")[1:])
    _start = _model_wo_id.split("-f")[0]
    _end = _model_wo_id.split("-v")[-1]
    models = [model for model in models if _start in model and model.endswith(_end)]

    model_dict = {
        int(model.split("-f")[-1].split("-")[0]): model
        for model in models
    }
    model_dict = dict(sorted(model_dict.items(), key=lambda item: item[0]))
    # print(model_dict)

    # 2. get data from all models
    model_data = {k: _get_data(v) for k, v in model_dict.items()}
    # print(model_data)

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
