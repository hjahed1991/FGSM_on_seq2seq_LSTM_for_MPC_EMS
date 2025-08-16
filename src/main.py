import subprocess
import itertools
from tqdm import tqdm
import pathlib
# In this file, we will automate the process of running the model with different configurations.

seeds = [42,]#, 128, 256, 512, 1024, 2048
targets = ['consumption', 'pv_production' , 'wind_production'] # 
epsilons = [  0.1,  0.2,  0.3, 0.4, 0.5,] #
poisoning_ratios = [ 0.2, 0.5, 1]
forecast_horizons = [ 24]#, 12, 18, 24]
# process_types = ['clean']#, 'poisoned']
models = ['LSTM']#, 'GRU', 'Transformer']
# poisoning_models = ['FGSM']#, 'PGD', 'CW']
sequence_lengths = [24]#, 36, 48]

# --- Determine script directory --- 
script_dir = pathlib.Path(__file__).parent.resolve()
print(f"Script directory: {script_dir}")

TUNE_HYPERPARAMETERS_FILE_PATH = script_dir / 'tune_hyperparameters.py'
TRAIN_MODEL_FILE_PATH = script_dir / 'train_model.py'
EVALUATE_MODEL_FILE_PATH = script_dir / 'evaluate_model.py'
HISTORICAL_FORECAST_FILE_PATH = script_dir / 'historical_forecast.py'
GENERATE_ADVERSARIAL_SAMPLES_FILE_PATH = script_dir / 'generate_adversarial_samples.py'
TRAIN_POISONED_MODEL_FILE_PATH = script_dir / 'train_poisoned_model.py'
EVALUATE_POISONED_MODEL_FILE_PATH = script_dir / 'evaluate_poisoned_model.py'
POISONED_HISTORICAL_FORECAST_FILE_PATH = script_dir / 'poisoned_historical_forecast.py'


outer_combinations = list(itertools.product(targets, models, forecast_horizons, sequence_lengths))

for target, model, horizon, sequence_length in tqdm(outer_combinations, desc="Overall Progress", dynamic_ncols=True):
    # TODO: Implement or uncomment hyperparameter tuning if needed
    tqdm.write(f"Running hyperparameter tuning for Target={target}, Model={model}, Horizon={horizon}, SequenceLength={sequence_length}")
    subprocess.run([
        'python', TUNE_HYPERPARAMETERS_FILE_PATH,
        '--target', target,
        '--model', model,
        '--horizon', str(horizon),
        '--sequence_length', str(sequence_length)
    ], check=True, text=True)

    for seed in tqdm(seeds, desc=f"Seed (T:{target}, M:{model}, H:{horizon}, SL:{sequence_length})", leave=False, dynamic_ncols=True):
        tqdm.write(f"--- Processing: Target={target}, Model={model}, Horizon={horizon}, Seed={seed} ---")
        # Training clean model:
        tqdm.write("Running: Training Clean Model")
        subprocess.run([
            'python', TRAIN_MODEL_FILE_PATH,
            '--target', target,
            '--model', model,
            '--horizon', str(horizon),
            '--sequence_length', str(sequence_length),
            '--seed', str(seed)
        ], check=True, text=True)
        tqdm.write("Running: Evaluating Clean Model")
        subprocess.run([
            'python', EVALUATE_MODEL_FILE_PATH,
            '--target', target,
            '--model', model,
            '--horizon', str(horizon),
            '--sequence_length', str(sequence_length),
            '--seed', str(seed),
        ], check=True, text=True)
        tqdm.write("Running: Historical Forecast (Clean)")
        subprocess.run([
            'python', HISTORICAL_FORECAST_FILE_PATH,
            '--target', target,
            '--model', model,
            '--horizon', str(horizon),
            '--sequence_length', str(sequence_length),
            '--seed', str(seed)
        ], check=True, text=True)

        # Training poisoned model:
        inner_poison_combinations = list(itertools.product(epsilons, poisoning_ratios))
        for epsilon, poisoning_ratio in tqdm(inner_poison_combinations, desc=f"Poison Params (Seed:{seed})", leave=False, dynamic_ncols=True):
            tqdm.write(f"Running: FGSM - Epsilon={epsilon}, Ratio={poisoning_ratio}")
            tqdm.write("Running: Generate Adversarial Samples")
            subprocess.run([
                'python', GENERATE_ADVERSARIAL_SAMPLES_FILE_PATH,
                '--target', target,
                '--model', model, # Assuming base model type is needed
                '--horizon', str(horizon),
                '--sequence_length', str(sequence_length),
                '--seed', str(seed),
                '--epsilon', str(epsilon),
                '--poisoning_ratio', str(poisoning_ratio)
            ], check=True, text=True)
            tqdm.write(f"Running: Train Poisoned Model - Epsilon={epsilon}, Ratio={poisoning_ratio}")
            subprocess.run([
                'python', TRAIN_POISONED_MODEL_FILE_PATH,
                '--target', target,
                '--model', model,
                '--horizon', str(horizon),
                '--sequence_length', str(sequence_length),
                '--seed', str(seed),
                '--epsilon', str(epsilon),
                '--poisoning_ratio', str(poisoning_ratio)
            ], check=True, text=True)
            tqdm.write(f"Running: Evaluate Poisoned Model - Epsilon={epsilon}, Ratio={poisoning_ratio}")
            subprocess.run([
                'python', EVALUATE_POISONED_MODEL_FILE_PATH,
                '--target', target,
                '--model', model,
                '--horizon', str(horizon),
                '--sequence_length', str(sequence_length),
                '--seed', str(seed),
                '--epsilon', str(epsilon),
                '--poisoning_ratio', str(poisoning_ratio)
            ], check=True, text=True)
            tqdm.write(f"Running: Historical Forecast (Poisoned) - Epsilon={epsilon}, Ratio={poisoning_ratio}")
            subprocess.run([
                'python', POISONED_HISTORICAL_FORECAST_FILE_PATH,
                '--target', target,
                '--model', model,
                '--horizon', str(horizon),
                '--sequence_length', str(sequence_length),
                '--seed', str(seed),
                '--epsilon', str(epsilon),
                '--poisoning_ratio', str(poisoning_ratio)
            ], check=True, text=True)
                    
                  

tqdm.write("--- Automated Simulation completed ---")



