from env.pendulum_env import PendulumEnv
from stable_baselines3 import PPO

model = PPO.load("ppo_pendulum")

env = PendulumEnv(
    xml_file="model/pendulum.xml",
    render_mode="human",   # 描画
    reset_noise_scale=0.1, # 評価時はノイズなしが見やすい
    theta_threshold=0.2,
)

obs, info = env.reset(seed=0)

for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()

env.close()
