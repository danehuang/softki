import argparse
from omegaconf import OmegaConf

import gp.svi_gp
import gp.svi_gp.svi_gp
import gp.sv_gp.sv_gp
import gp.soft_gp.train
from gp.util import *


if __name__ == "__main__":
    from data.get_uci import (
        PoleteleDataset,
        ElevatorsDataset,
        BikeDataset,
        Kin40KDataset,
        ProteinDataset,
        KeggDirectedDataset,
        CTSlicesDataset,
        KeggUndirectedDataset,
        RoadDataset,
        SongDataset,
        BuzzDataset,
        HouseElectricDataset,
    )
    
    # Create config
    config = OmegaConf.merge(gp.svi_gp.svi_gp.CONFIG, gp.sv_gp.sv_gp.CONFIG, gp.soft_gp.train.CONFIG)

    # Omega config to argparse
    parser = argparse.ArgumentParser(description="Example of converting OmegaConf to argparse")
    parser.add_argument("--data_dir", type=str, default="data/uci_datasets/uci_datasets")
    for key, value in flatten_dict(OmegaConf.to_container(config, resolve=True)).items():
        arg_type = type(value)  # Infer the type from the configuration
        parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'Default: {value}')
    args = parser.parse_args()
    cli_config = OmegaConf.create(unflatten_dict(vars(args)))
    config = OmegaConf.merge(config, cli_config)

    if config.model.name == "svi-gp":
        train_gp = gp.svi_gp.svi_gp.train_gp
    elif config.model.name == "soft-gp":
        train_gp = gp.soft_gp.train.train_gp
    elif config.model.name == "sv-gp":
        train_gp = gp.sv_gp.sv_gp.train_gp
    else:
        raise ValueError(f"Name not found {config.model.name}")

    # Create dataset
    if config.dataset.name == "pol":
        dataset = PoleteleDataset(f"{args.data_dir}/pol/data.csv")
    elif config.dataset.name == "elevators":
        dataset = ElevatorsDataset(f"{args.data_dir}/elevators/data.csv")
    elif config.dataset.name == "bike":
        dataset = BikeDataset(f"{args.data_dir}/bike/data.csv")
    elif config.dataset.name == "kin40k":
        dataset = Kin40KDataset(f"{args.data_dir}/kin40k/data.csv")
    elif config.dataset.name == "protein":
        dataset = ProteinDataset(f"{args.data_dir}/protein/data.csv")
    elif config.dataset.name == "keggdirected":
        dataset = KeggDirectedDataset(f"{args.data_dir}/keggdirected/data.csv")
    elif config.dataset.name == "slice":
        dataset = CTSlicesDataset(f"{args.data_dir}/slice/data.csv")
    elif config.dataset.name == "keggundirected":
        dataset = KeggUndirectedDataset(f"{args.data_dir}/keggundirected/data.csv")
    elif config.dataset.name == "3droad":
        dataset = RoadDataset(f"{args.data_dir}/3droad/data.csv")
    elif config.dataset.name == "song":
        dataset = SongDataset(f"{args.data_dir}/song/data.csv")
    elif config.dataset.name == "buzz":
        dataset = BuzzDataset(f"{args.data_dir}/buzz/data.csv")
    elif config.dataset.name == "houseelectric":
        dataset = HouseElectricDataset(f"{args.data_dir}/houseelectric/data.csv")
    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported ...")
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_frac=config.dataset.train_frac,
        val_frac=config.dataset.val_frac
    )

    # Train
    model = train_gp(config, train_dataset, test_dataset)
    