"""
train.py — RF-DETR model configuration and training orchestration.
"""

from rfdetr.config import TrainConfig, RFDETRSmallConfig
from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer
from pathlib import Path

# model configuration
def build_model_config(config: dict) -> RFDETRSmallConfig:
    return RFDETRSmallConfig(
        num_classes=int(config["num_classes"]),
        resolution=int(config["resolution"]),
    )


# training configuration
def build_train_config(config: dict) -> TrainConfig:
    train = config["train"]
    data = config["data"]
    validation = config["validation"]
    output = config["output"]
    hardware = config["hardware"]

    return TrainConfig(
        dataset_file=data["dataset_file"],
        dataset_dir=data["dataset_dir"],
        epochs=train["epochs"],
        batch_size=train["batch_size"],
        grad_accum_steps=train["grad_accum_steps"],
        lr=train["lr"],
        lr_encoder=train["lr_encoder"],
        weight_decay=train["weight_decay"],
        seed=train["seed"],
        eval_interval=validation["eval_interval"],
        best_model_metric=validation["best_model_metric"],
        output_dir=output["output_dir"],
        checkpoint_interval=output["checkpoint_interval"],
        accelerator=hardware["accelerator"],
        devices=hardware["devices"],
        num_workers=hardware["num_workers"],
        logging = config["logging"]
        wandb=logging["wandb"],
        mlflow=logging["mlflow"],
        tensorboard=logging["tensorboard"],
        project=logging["project"],
        run=logging["run"],
    )

# prepare COCO dataset layout for RF-DETR
def prepare_coco_layout(data_config: dict) -> None:

    root = Path(data_config["dataset_dir"])
    aliases = [
        (root / data_config["train_images"], root / "train2017"),
        (root / data_config["val_images"], root / "val2017"),
        (root / data_config["train_annotations"], root / "annotations" / "instances_train2017.json",),
        (root / data_config["val_annotations"], root / "annotations" / "instances_val2017.json",),
    ]

    for source, destination in aliases:
        if not source.exists():
            raise FileNotFoundError(f"Dataset path not found: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            continue

        destination.symlink_to(source.resolve())

    print("COCO dataset paths prepared for RF-DETR.")

# build training components: model, datamodule, trainer
def build_training_components( model_config, train_config, extra_callbacks=None, trainer_kwargs=None,):

    model = RFDETRModelModule(model_config=model_config, train_config=train_config,)

    datamodule = RFDETRDataModule(model_config=model_config, train_config=train_config,)

    trainer = build_trainer(
        train_config,
        model_config,
        extra_callbacks=extra_callbacks or [],
        **(trainer_kwargs or {}),
    )

    return model, datamodule, trainer

# run training loop
def run_training( model_config, train_config, data_config, extra_callbacks=None, trainer_kwargs=None,):

    prepare_coco_layout(data_config)

    model, datamodule, trainer = build_training_components(
        model_config=model_config,
        train_config=train_config,
        extra_callbacks=extra_callbacks,
        trainer_kwargs=trainer_kwargs,
    )

    trainer.fit( model, datamodule, ckpt_path=train_config.resume or None,)

    return trainer