# finetune.py : Entry point for RF-DETR tent detection fine-tuning.


import yaml

from tentdet_hot.train import build_model_config, build_train_config, run_training

def main():
    # Load YAML configuration.
    with open("configs/train.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Build configurations.
    model_config = build_model_config(config["model"])
    train_config = build_train_config(config)

    # Run training.
    run_training(
        model_config=model_config,
        train_config=train_config,
        data_config=config["data"],
    )


if __name__ == "__main__":
    main()