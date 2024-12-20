# Arguments
import argparse
from omegaconf import OmegaConf

# Our imports
import gp.svgp.train
import gp.sgpr.train
import gp.softki.train
import gp.exact.train
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
    from data.get_md22 import (
        MD22_AcAla3NHME_Dataset,
        MD22_DHA_Dataset,
        MD22_DNA_AT_AT_CG_CG_Dataset,
        MD22_DNA_AT_AT_Dataset,
        MD22_Stachyose_Dataset,
        MD22_Buckyball_Catcher_Dataset,
        MD22_DoubleWalledNanotube_Dataset,
    )

    # Omega config to argparse
    config = OmegaConf.merge(gp.svgp.train.CONFIG, gp.sgpr.train.CONFIG, gp.softki.train.CONFIG)
    parser = argparse.ArgumentParser(description="Example of converting OmegaConf to argparse")
    parser.add_argument("--data_dir", type=str, default="data/uci_datasets/uci_datasets")
    for key, value in flatten_dict(OmegaConf.to_container(config, resolve=True)).items():
        arg_type = type(value)  # Infer the type from the configuration
        if key == "model.name":
            parser.add_argument(f'--{key}', type=arg_type, default="soft-gp")
        elif arg_type == bool:
            parser.add_argument(f'--{key}', action='store_true')
        else:
            parser.add_argument(f'--{key}', type=arg_type)
    args = parser.parse_args()
    cli_config = vars(args)

    def merge_dicts_keep_latest_not_none(dict1, dict2):
        merged_dict = dict1.copy()  # Start with a copy of dict1

        for key, value in dict2.items():
            # Only update if value is not None
            if value is not None:
                merged_dict[key] = value

        return merged_dict

    # Config and train function factory
    if cli_config["model.name"] == "svgp":
        train_gp = gp.svgp.train.train_gp
        config = OmegaConf.create(unflatten_dict(flatten_omegaconf(merge_dicts_keep_latest_not_none(gp.svgp.train.CONFIG, cli_config))))
    elif cli_config["model.name"] == "softki":
        train_gp = gp.softki.train.train_gp
        config = OmegaConf.create(unflatten_dict(flatten_omegaconf(merge_dicts_keep_latest_not_none(gp.softki.train.CONFIG, cli_config))))
    elif cli_config["model.name"] == "sgpr":
        train_gp = gp.sgpr.train.train_gp
        config = OmegaConf.create(unflatten_dict(flatten_omegaconf(merge_dicts_keep_latest_not_none(gp.sgpr.train.CONFIG, cli_config))))
    elif cli_config["model.name"] == "exact":
        train_gp = gp.exact.train.train_gp
        config = OmegaConf.create(unflatten_dict(flatten_omegaconf(merge_dicts_keep_latest_not_none(gp.exact.train.CONFIG, cli_config))))
    else:
        raise ValueError(f"Name not found {config.model.name}")

    # Dataset factory
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
    elif config.dataset.name == "Ac-Ala3-NHMe":
        dataset = MD22_AcAla3NHME_Dataset(f"{args.data_dir}/md22_Ac-Ala3-NHMe.npz")
    elif config.dataset.name == "AT-AT":
        dataset = MD22_DNA_AT_AT_Dataset(f"{args.data_dir}/md22_AT-AT.npz")
    elif config.dataset.name == "AT-AT-CG-CG":
        dataset = MD22_DNA_AT_AT_CG_CG_Dataset(f"{args.data_dir}/md22_AT-AT-CG-CG.npz")
    elif config.dataset.name == "stachyose":
        dataset = MD22_Stachyose_Dataset(f"{args.data_dir}/md22_stachyose.npz")
    elif config.dataset.name == "DHA":
        dataset = MD22_DHA_Dataset(f"{args.data_dir}/md22_DHA.npz")
    elif config.dataset.name == "buckyball-catcher":
        dataset = MD22_Buckyball_Catcher_Dataset(f"{args.data_dir}/md22_buckyball-catcher.npz")
    elif config.dataset.name == "double-walled-nanotube":
        dataset = MD22_DoubleWalledNanotube_Dataset(f"{args.data_dir}/md22_double-walled_nanotube.npz")
    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported ...")
    
    # Seed
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    # Generate splits
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_frac=config.dataset.train_frac,
        val_frac=config.dataset.val_frac
    )

    # Train
    model = train_gp(config, train_dataset, test_dataset)
    