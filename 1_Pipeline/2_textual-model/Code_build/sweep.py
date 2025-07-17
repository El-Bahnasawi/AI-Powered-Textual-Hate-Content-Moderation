import wandb

sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'eval_matthews_corrcoef',
        'goal': 'maximize'   
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3
    },
    'parameters': {
        'lora_r': {
            'values': [4, 8, 16, 32]  # Smaller r for bayes
        },
        'lora_alpha': {
            'distribution': 'int_uniform',
            'min': 4,
            'max': 32
        },
        'lora_dropout': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.2
        },
        'learning_rate': {
        'distribution': 'log_uniform',
        'min': 1e-6,  # Lower minimum (was 1e-5)
        'max': 5e-5   # More conservative maximum (was 1e-3)
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="distilbert-lora-bayes")