# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
from pathlib import Path

import lightning as L
from omegaconf import OmegaConf

from enhancing.utils.general import get_config_from_file, initialize_from_config, setup_callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-a", "--accelerator", type=str, default="gpu")
    parser.add_argument("-nd", "--num_devices", type=int, default=1)
    parser.add_argument("-u", "--gradient_steps", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-lr", "--base_lr", type=float, default=4.5e-6)
    parser.add_argument("-a", "--use_amp", default=False, action="store_true")
    parser.add_argument("-b", "--batch_frequency", type=int, default=750)
    parser.add_argument("-m", "--max_images", type=int, default=4)
    args = parser.parse_args()

    # Set random seed
    L.seed_everything(args.seed)

    # Load configuration
    config = get_config_from_file(Path.cwd().joinpath("configs", args.config + ".yaml"))
    exp_config = OmegaConf.create(
        {
            "name": args.config,
            "epochs": args.epochs,
            "update_every": args.gradient_steps,
            "base_lr": args.base_lr,
            "use_amp": args.use_amp,
            "batch_frequency": args.batch_frequency,
            "max_images": args.max_images,
        }
    )

    # Build model
    model = initialize_from_config(config.model)
    model.learning_rate = exp_config.base_lr

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config)

    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()

    # Build trainer
    trainer = L.Trainer(
        max_epochs=exp_config.epochs,
        precision=16 if exp_config.use_amp else 32,
        callbacks=callbacks,
        strategy="ddp" if args.num_devices > 1 else None,
        accumulate_grad_batches=exp_config.update_every,
        logger=logger,
    )

    # Train
    trainer.fit(model, data)
