import math
import logging

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import wandb
from easydict import EasyDict

from src.datasets.datamodule import PANORAMIADataModule

def fine_tune_generator(
    config: EasyDict, 
    dm: PANORAMIADataModule,
    ):
    # load training and validation datasets from data module
    train_dataset, validation_dataset, _ = dm.get_generator_training_datasets()

    # load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(config.generator.train.pretrained_model_name_or_path)

    # loading optimization hyperparameters from the config file
    opt_hyp_paramrs = config.generator.train.optimization

    # setting the training arguments
    training_args = TrainingArguments(
        # report_to = 'wandb',
        # run_name = config['train']['exp_name'],
        output_dir=config.generator.train.saving_dir,
        num_train_epochs=opt_hyp_paramrs.epoch,
        evaluation_strategy="epoch",
        learning_rate=opt_hyp_paramrs.learning_rate,
        weight_decay=opt_hyp_paramrs.weight_decay,
        warmup_steps=opt_hyp_paramrs.warmup_steps,
        per_device_train_batch_size=opt_hyp_paramrs.per_device_batch_size,
        
        save_strategy="no",
        do_train=True,
        do_eval=True
    )

    # initializing wandb for visualization
    wandb.init(
            project=config.base.project_name,
            group="generator-fine-tune",
            name=config.generator.train.run_name,
            config=training_args
        )
    
    logging.info(f"Fine-tuning the generator with hyperparameters:\n{training_args}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # fine-tune the generator
    trainer.train()

    # saving the final model
    model.save_pretrained(config.generator.train.saving_dir)

    
    eval_results = trainer.evaluate()
    print(eval_results)
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")