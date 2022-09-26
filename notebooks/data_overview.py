
import pandas as pd
from typing import List
import matplotlib.pyplot as plt


def zero_total(_df):
    drop_total(_df)
    _df["total"] = 0
    _df.loc["total"] = 0


def drop_total(_df):
    _df.drop("total", axis=0, inplace=True)
    _df.drop("total", axis=1, inplace=True)


def add_total(_df):
    _df["total"] = _df.apply(lambda x: sum(x), axis=1)
    _df.loc["total"] = _df.apply(lambda x: sum(x), axis=0)


def totex(_df, _name, header, tail="\end{tabular}}"):
    t = "\\scalebox{\\tabscale}{"
    t += header + " \n"
    t += _df.to_csv().replace(
        ",", " & ").replace(
        "commoncrawl", "cc").replace(
        "conversational", "conv").replace(
        "\n", " \\\\ \n").replace(
        "_", "\_").replace(
        "\\\\", "\\\\ \\hline", 1
    )
    if t.endswith(" \\\\ \n"):
        t = t[:-len(" \\\\ \n")]
        t += " \n"
    t = "\\\\ \\hline".join(t.rsplit("\\\\", 1))  # replace last "\\" by "\\ \hline"
    t += tail

    path = f"tables/{_name}.tex"
    with open(path, "w") as f:
        f.write(t)


def get_dfs(_dijtokens, _Eij):
    idx = list(_Eij["en"].keys())

    _dfs = dict()

    _dfs["dijtokens"] = pd.DataFrame(_dijtokens)
    _dfs["dijtokens"] = _dfs["dijtokens"].reindex(idx)
    _dfs["dijtokens"] *= 10 ** 9
    add_total(_dfs["dijtokens"])

    _dfs["Eij"] = pd.DataFrame(_Eij)
    _dfs["Eij"] = _dfs["Eij"].reindex(idx)
    add_total(_dfs["Eij"])
    zero_total(_dfs["Eij"])

    return _dfs


def get_unique_repeated_discarded(_used, _existing):
    _unique_lists = [[min(c,d) for c, d in zip(a, b)] for a, b in zip(_used.values, _existing.values)]
    _repeated_lists = [[max(0,c-d) for c, d in zip(a, b)] for a, b in zip(_used.values, _existing.values)]
    _discarded_lists = [[max(0,d-c) for c, d in zip(a, b)] for a, b in zip(_used.values, _existing.values)]
    return _unique_lists, _repeated_lists, _discarded_lists


def get_total(_dfs, _factors):
    f_unique_total = {f: 0 for f in _factors}
    f_repeated_total = {f: 0 for f in _factors}
    f_discarded_total = {f: 0 for f in _factors}
    f_used_total = {f: 0 for f in _factors}

    idx = _dfs["dijtokens"].index
    cols = _dfs["dijtokens"].columns

    for f in _factors:
        f_used = f * _dfs["Eij"] * _dfs["dijtokens"]
        f_existing = _dfs["dijtokens"]
        f_unique_lists, f_repeated_lists, f_discarded_lists = get_unique_repeated_discarded(f_used, f_existing)

        f_unique = pd.DataFrame(f_unique_lists,
                                columns=cols,
                                index=idx)
        f_repeated = pd.DataFrame(f_repeated_lists,
                                  columns=cols,
                                  index=idx)
        f_discarded = pd.DataFrame(f_discarded_lists,
                                   columns=cols,
                                   index=idx)
        drop_total(f_unique)
        drop_total(f_repeated)
        drop_total(f_discarded)
        f_unique_total[f] += f_unique.sum().sum() / 10 ** 9
        f_repeated_total[f] += f_repeated.sum().sum() / 10 ** 9
        f_discarded_total[f] += f_discarded.sum().sum() / 10 ** 9
        f_used_total[f] += f_unique.sum().sum() / 10 ** 9 + f_repeated.sum().sum() / 10 ** 9

    # f_unique_total, f_repeated_total, f_discarded_total, f_used_total

    unique_total = list(f_unique_total.values())
    repeated_total = list(f_repeated_total.values())
    discarded_total = list(f_repeated_total.values())

    return unique_total, repeated_total, discarded_total


def plot_data_distribution(_factors, _T, _Tthr, _unique_total, _repeated_total, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if not isinstance(ax, list):
        ax = [ax, None]

    factorsT = [elem * _T / 10 ** 9 for elem in _factors]
    unique_percent = [u / (u + r) * 100 for u, r in zip(_unique_total, _repeated_total)]
    # _ = ax[0].plot(factorsT, unique_total, marker="o", linestyle="", color="green", label="unique")
    # _ = ax[0].plot(factorsT, [a+b for a,b in zip(unique_total, repeated_total)], marker="o", linestyle="", color="orange", label="repeated")
    _ = ax[0].fill_between(factorsT, 0, _unique_total, color="green", alpha=0.5, label="unique")
    _ = ax[0].fill_between(factorsT, _unique_total, [a + b for a, b in zip(_unique_total, _repeated_total)],
                           color="orange", alpha=0.5, label="repeated")
    ax[0].set_xlim([0, _T / 10 ** 9])
    ax[0].set_ylim([0, _T / 10 ** 9])
    ax[0].set_xlabel("t [10^9 tokens]")
    ax[0].set_ylabel("t [10^9 tokens]")
    if title is not None:
        _ = ax[0].set_title(title)
    _ = ax[0].legend(loc="upper left")

    ax2 = ax[0].twinx()
    ax2.plot(factorsT, unique_percent, color="green", label="unique percentage")
    ax2.set_ylim([0, 100])
    if _Tthr is not None:
        ax2.plot([_Tthr / 10 ** 9, _Tthr / 10 ** 9], [0, 100], linestyle=":", color="k", label="T_thr")
    _ = ax2.set_ylabel("unique percentage [%]")
    _ = ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig


class DataBase:

    def __init__(self, title: str, factors: List[float], Tthr: float = None):
        self.T = None
        self.dijtokens = None
        self.Eij = None
        self.dfs = None
        self.unique_total, self.repeated_total, self.discarded_total = None, None, None
        self.title = title
        self.factors = factors
        self.Tthr = Tthr

    def process(self):
        self.T = sum([a*b for a, b in zip(self.dijtokens["en"].values(), self.Eij["en"].values())])*10**9
        self.dfs = get_dfs(self.dijtokens, self.Eij)
        self.unique_total, self.repeated_total, self.discarded_total = get_total(self.dfs, self.factors)

    def plot_data_distribution(self):
        plot_data_distribution(self.factors, self.T, self.Tthr, self.unique_total, self.repeated_total, self.title)


class DataGPT3(DataBase):

    def __init__(self, title: str, factors: List[float], Tthr: float = None):
        super().__init__(title, factors, Tthr)
        self.T_check = 300*10**9
        self.dijtokens = {"en": {"cc": 410, "webtext2": 19, "books1": 12, "books2": 55, "wikipedia": 3}}
        self.Eij = {"en": {"cc": 0.44, "webtext2": 3.4, "books1": 1.9, "books2": 0.43, "wikipedia": 2.9}}
        self.process()


class DataChinchilla(DataBase):

    def __init__(self, title: str, factors: List[float], Tthr: float = None):
        super().__init__(title, factors, Tthr)
        self.T_check = 1400*10**9
        self.dijtokens = {"en": {"massiveweb": 508, "books": 560, "c4": 182, "news": 667, "github": 430, "wikipedia": 4}}
        self.Eij = {"en": {"massiveweb": 1.24, "books": 0.75, "c4": 0.77, "news": 0.21, "github": 0.13, "wikipedia": 3.40}}
        self.process()


class DataThePile(DataBase):

    def __init__(self, title: str, factors: List[float], Tthr: float = None):
        super().__init__(title, factors, Tthr)
        self.T_check = 0.29335*1254.20*10**9  # tokens_per_byte across whole dataset / total pile dataset size in bytes

        self.dijbytes = {
            "Pile-CC": 227.12,
            "PubMed Central": 90.27,
            "Books3": 100.96,
            "OpenWebText2": 62.77,
            "Arxiv": 56.21,
            "Github": 95.16,
            "FreeLaw": 51.15,
            "StackExchange": 32.30,
            "USPTO Backgrounds": 22.90,
            "PubMed Abstracts": 19.26,
            "Gutenberg(PG - 19)": 10.88,
            "OpenSubtitles": 12.98,
            "Wikipedia(en)": 6.38,
            "DM Mathematics": 7.75,
            "Ubuntu IRC": 5.52,
            "BookCorpus2": 6.30,
            "EuroParl": 4.59,
            "HackerNews": 3.90,
            "YoutubeSubtitles": 3.73,
            "PhilPapers": 2.38,
            "NIH ExPorter": 1.89,
            "Enron Emails": 0.88,
        }

        self.tokens_per_bytes = {
            "Pile-CC": 0.2291,
            "PubMed Central": 0.3103,
            "Books3": 0.2477,
            "OpenWebText2": 0.2434,
            "Arxiv": 0.3532,
            "Github": 0.4412,
            "FreeLaw": 0.2622,
            "StackExchange": 0.3436,
            "USPTO Backgrounds": 0.2116,
            "PubMed Abstracts": 0.2183,
            "Gutenberg(PG - 19)": 0.2677,
            "OpenSubtitles": 0.2765,
            "Wikipedia(en)": 0.2373,
            "DM Mathematics": 0.8137,
            "Ubuntu IRC": 0.3651,
            "BookCorpus2": 0.2430,
            "EuroParl": 0.3879,
            "HackerNews": 0.2627,
            "YoutubeSubtitles": 0.4349,
            "PhilPapers": 0.2688,
            "NIH ExPorter": 0.1987,
            "Enron Emails": 0.3103,
        }

        self.dijtokens = {"en": {k: v*self.tokens_per_bytes[k] for k, v in self.dijbytes.items()}}

        self.Eij = {"en": {
            "Pile-CC": 1.0,
            "PubMed Central": 2.0,
            "Books3": 1.5,
            "OpenWebText2": 2.0,
            "Arxiv": 2.0,
            "Github": 1.0,
            "FreeLaw": 1.5,
            "StackExchange": 2.0,
            "USPTO Backgrounds": 2.0,
            "PubMed Abstracts": 2.0,
            "Gutenberg(PG - 19)": 2.5,
            "OpenSubtitles": 1.5,
            "Wikipedia(en)": 3.0,
            "DM Mathematics": 2.0,
            "Ubuntu IRC": 2.0,
            "BookCorpus2": 1.5,
            "EuroParl": 2.0,
            "HackerNews": 2.0,
            "YoutubeSubtitles": 2.0,
            "PhilPapers": 2.0,
            "NIH ExPorter": 2.0,
            "Enron Emails": 2.0,
        }}
        self.process()
