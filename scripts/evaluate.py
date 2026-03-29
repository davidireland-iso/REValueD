"""Evaluation script for trained models."""
import argparse
from pathlib import Path
import yaml
import numpy as np
import wandb

from loguru import logger

from revalued.algorithms import DecQN, REValueD, ActionConditionedDecQN
from revalued.utils import set_seeds, make_env


def load_model(model_path: Path, config_path: Path):
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file

    Returns:
        Loaded algorithm
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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

    # Load weights
    algorithm.load(model_path)

    return algorithm, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for evaluation'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes (if supported)'
    )
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save videos of episodes'
    )
    parser.add_argument(
        '--log-wandb',
        action='store_true',
        help='Log evaluation results to Weights & Biases'
    )

    args = parser.parse_args()

    # Set seeds
    set_seeds(args.seed)

    # Load model
    logger.info(f"Loading model from {args.model}")
    algorithm, config = load_model(args.model, args.config)

    # Create environment
    env = make_env(
        config['environment']['domain'],
        config['environment']['task'],
        config['environment'].get('bin_size', 3),
        config['environment'].get('factorised', True),
        seed=args.seed
    )

    # Run evaluation
    logger.info(f"Running {args.episodes} evaluation episodes")

    scores = []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        score = 0.0
        steps = 0

        while not done:
            action = algorithm.greedy_act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1

            if args.render:
                env.render()

        scores.append(score)
        logger.info(f"Episode {episode + 1}: Score = {score:.2f}, Steps = {steps}")

    # Report results
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}")
    logger.info(f"Min Score: {min_score:.2f}")
    logger.info(f"Max Score: {max_score:.2f}")
    logger.info("=" * 50)

    if args.log_wandb:
        wandb.init(
            project="action-conditioning",
            config=config,
            name=f"eval_{config['experiment']['name']}_seed_{args.seed}_inference_run",
            tags=["inference"],
        )
        wandb.log({
            "eval/score_mean": mean_score,
            "eval/score_std": std_score,
            "eval/score_min": min_score,
            "eval/score_max": max_score,
            "eval/episodes": args.episodes,
            "eval/seed": args.seed,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
