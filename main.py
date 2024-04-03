import argparse

from runner.base import BaseExperiment
from runner.unconditional import UnconditionalExperiment
from tools.config_utils import init_experiment_config, override_phase_config, parser_override_args

experiments = {
    "base": BaseExperiment,
    "unconditional": UnconditionalExperiment,
}


def parser_args():
    parser = argparse.ArgumentParser("Diffusion config")

    parser.add_argument("--config", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--phase",
        "-p",
        type=str,
        required=True,
        choices=["train", "inference", "sample"],
    )

    args, kwargs = parser.parse_known_args()

    return args, kwargs


def main():
    args, kwargs = parser_args()
    config = init_experiment_config(args.config)
    # update phase config
    config.update({"phase": args.phase})

    # update kwargs
    # the 'a.b=True' to {'a': {'b': True}}
    config = override_phase_config(config)
    config = parser_override_args(config, kwargs)

    experiment_cls = experiments[config.experiment_name]
    assert experiment_cls is not None, f"Experiment {config.experiment_name} not found"

    #  create experiment instance
    experiment_instance = experiment_cls(config)
    assert experiment_instance is not None, f"Experiment {config.experiment_name} not initialized"

    phase = config.phase
    if phase == "train":
        experiment_instance.train()

    elif phase == "inference":
        experiment_instance.inference()

    elif phase == "sample":
        experiment_instance.sample()

    elif phase == "test":
        print("The test phase is not implemented in the current code. Please use an external script to test the model.")
        print("the image test script is in the evaluations folder")


if __name__ == "__main__":
    main()
