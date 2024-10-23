import numpy as np
from tqdm import tqdm
import wandb


class Experiment:
    def __init__(self, run):
        self.name = run.name
        self.config = run.config
        self.summary = run.summary
        self.history = run.history()
        self.tags = run.tags
        self.run = run
        
    def get_history(self):
        return np.array(list(self.history['additional_steps'])).cumsum()


def fetch(project, filters):
    api = wandb.Api()
    entity = "bogp"
    hdata = []
    runs = api.runs(entity + "/" + project, filters=filters)
    for run in tqdm(runs):
        try:
            hdata.append(Experiment(run))
        except:
            pass
    return hdata


def init_uci_dict():
    return {
        "pol": {},
        "elevators": {},
        "bike": {},
        "kin40k": {},
        "protein": {},
        "keggdirected": {},
        "slice": {},
        "keggundirected": {},
        "3droad": {},
        "song": {},
        "buzz": {},
        "houseelectric": {},
    }

uci_markers = {
    "pol": "4",
    "elevators": "x",
    "bike": "*",
    "kin40k": "o",
    "protein": "d",
    "keggdirected": ".",
    "slice": "p",
    "keggundirected": "v",
    "3droad": "1",
    "song": "s",
    "buzz": "2",
    "houseelectric": "3",
}


UCI_TEST_RMSE_LIMITS = {
    "pol": (0.2, 0.35),
    "elevators": (0.38, 0.6),
    "bike": (0.18 , 0.45),
    "kin40k": (0.2, 0.75),
    "protein": (0.65, 0.85),
    "keggdirected": (0.08, 0.2),
    "slice": (0.04, 0.3),
    "keggundirected": (0.1, 0.3),
    "3droad": (0.6, 0.8),
    "song": (0.79, 0.85),
    "buzz": (0.245, 0.3),
    "houseelectric": (0.0, 0.2),
}


UCI_DATA_SIZE = {
    "pol": (15000, 26),
    "elevators": (16599, 18),
    "bike": (17379 , 17),
    "kin40k": (40000, 8),
    "protein": (45730, 9),
    "keggdirected": (48827, 20),
    "slice": (53500, 385),
    "keggundirected": (63608, 27),
    "3droad": (434874, 3),
    "song": (300000, 90),
    "buzz": (583250, 77),
    "houseelectric": (2049280, 11),
}

UCI_EXACT = {
    "pol": .151,
    "elevators": .394,
    "bike": .220,
    "kin40k": .099,
    "protein": .536,
    "keggdirected": .086,
    "slice": .262,
    "keggundirected": .118,
    "3droad": .101,
    "song": .807,
    "buzz": .288,
    "houseelectric": .055,
}


def get_uci_info():
    datasets = ["pol", "elevators", "bike", "kin40k", "protein", "keggdirected", "slice", "keggundirected", "3droad", "song", "buzz", "houseelectric"]
    uci_info = [(dataset, UCI_DATA_SIZE[dataset][0], UCI_DATA_SIZE[dataset][1], UCI_EXACT[dataset]) for dataset in datasets]
    return uci_info


LINE_STYLES = ["dotted", "dashed", "solid", "dotted", "dashed", "solid", "dotted", "dashed", "solid"]

LINE_WIDTHS = [1, 1, 1, 2, 2, 2, 3, 3, 3]

COLORS = ['b', 'g', 'r', 'c', 'y', 'm', 'k']


MD22_INFO = [
    ("Ac-Ala3-NHMe", 85109, 42 * 3),
    ("DHA", 69753, 56 * 3),
    ("stachyose", 27272, 87 * 3),
    ("AT-AT", 20001, 118 * 3),
    ("AT-AT-CG-CG", 10153, 118 * 3)
]
