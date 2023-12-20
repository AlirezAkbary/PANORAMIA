import math
import logging


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
    # Commented this later. Now, we leave the instantiation of the model to trainer, to ensure reproducibility (Since the LM Head would be initialized from scratch)
    # model = AutoModelForCausalLM.from_pretrained(config.generator.train.pretrained_model_name_or_path)

    # loading optimization hyperparameters from the config file
    opt_hyp_paramrs = config.generator.train.optimization

    # setting the training arguments
    training_args = TrainingArguments(
        # report_to = 'wandb',
        # run_name = config['train']['exp_name'],
        output_dir=config.generator.train.saving_dir,
        seed=config.generator.train.seed, # Ensuring Reproducibility
        num_train_epochs=opt_hyp_paramrs.epoch,
        learning_rate=opt_hyp_paramrs.learning_rate,
        weight_decay=opt_hyp_paramrs.weight_decay,
        warmup_steps=opt_hyp_paramrs.warmup_steps,
        per_device_train_batch_size=opt_hyp_paramrs.per_device_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
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
        model_init=lambda: AutoModelForCausalLM.from_pretrained(config.generator.train.pretrained_model_name_or_path),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # fine-tune the generator
    trainer.train()

    # saving the final model
    # Commented this later. Saving model is now assigned to trainer.
    # trainer.save_model()
    # model.save_pretrained(config.generator.train.saving_dir)

    
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results:\n{eval_results}")
    logging.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    return trainer.model