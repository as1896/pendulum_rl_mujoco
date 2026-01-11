import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.pendulum_env import PendulumEnv
import multiprocessing as mp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
XML_PATH = PROJECT_ROOT / "model" / "pendulum.xml"

def make_env():
    return PendulumEnv(
        xml_file=str(XML_PATH),
        reset_noise_scale=0.1,
        theta_threshold=0.2,
        frame_skip=2,
    )

def main():
    venv = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )
    model.learn(total_timesteps=300_000)
    model.save("ppo_pendulum")
    venv.close()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
