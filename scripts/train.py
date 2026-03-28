"""Training script for REValueD algorithms."""
import argparse
from pathlib import Path
import yaml

from loguru import logger

from revalued.algorithms import DecQN, REValueD, ActionConditionedDecQN
from revalued.trainers import Trainer
from revalued.utils import set_seeds, make_env


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not str(config_path).endswith(".yaml"):
        config_path = Path(str(config_path) + ".yaml")
    with open(Path('configs', 'experiments', config_path), 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train REValueD algorithms')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (overrides config)'
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=None,
        help='Directory to save results'
    )
    parser.add_argument(
        '--log-wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.device is not None:
        config['algorithm']['device'] = args.device

    # Set seeds
    seed = config['experiment']['seed']
    set_seeds(seed)

    # Create save directory
    if args.save_dir:
        save_dir = Path('experiments', args.save_dir)
    else:
        save_dir = Path('experiments') / config['experiment']['name'] / f'seed_{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup logging to file
    logger.add(save_dir / 'train.log')

    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Save directory: {save_dir}")

    # Create environment to get dimensions
    env = make_env(
        config['environment']['domain'],
        config['environment']['task'],
        config['environment'].get('bin_size', 3),
        config['environment'].get('factorised', True)
    )

    state_dim = env.observation_space.shape[0]
    action_space = env.action_space

    # Create algorithm
    algorithm_name = config['algorithm']['name']
    algorithm_config = config['algorithm']

    if algorithm_name == 'DecQN':
        algorithm = DecQN(
            state_dim=state_dim,
            action_space=action_space,
            **algorithm_config
        )
    elif algorithm_name == 'REValueD':
        algorithm = REValueD(
            state_dim=state_dim,
            action_space=action_space,
            **algorithm_config
        )
    elif algorithm_name == 'ActionConditionedDecQN':
        algorithm = ActionConditionedDecQN(
            state_dim=state_dim,
            action_space=action_space,
            **algorithm_config
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Create trainer and train
    trainer = Trainer(
        algorithm=algorithm,
        config=config,
        save_dir=save_dir,
        log_wandb=args.log_wandb
    )

    trainer.train()

    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
