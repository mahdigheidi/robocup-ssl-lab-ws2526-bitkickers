import os

import numpy as np
import torch


class TrainingCallback:
    def __init__(self, save_dir="model_snapshots", save_freq=100):
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def load_if_exists(self, policy):
        """
        Loads the latest checkpoint from self.save_dir into the agent if available.
        Returns (starting_episode, rewards).
        """
        files = os.listdir(self.save_dir)
        checkpoint_files = []
        for fname in files:
            if fname.startswith("ppo_ep_") and fname.endswith(".pt"):
                try:
                    ep_str = fname[len("ppo_ep_"):-len(".pt")]
                    ep_num = int(ep_str)
                    checkpoint_files.append((ep_num, fname))
                except ValueError:
                    continue
        if not checkpoint_files:
            print(f"[Callback] No checkpoint found in {self.save_dir}. Starting fresh.")
            return 0, []
        # Find the checkpoint with the highest episode number
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        latest_ep, latest_fname = checkpoint_files[0]
        latest_path = os.path.join(self.save_dir, latest_fname)
        checkpoint = torch.load(latest_path, weights_only=False)
        policy.load_state_dict(checkpoint["policy"])
        rewards = checkpoint.get("rewards", [])
        print(f"[Callback] Resuming from episode {latest_ep} using checkpoint {latest_fname}")
        return latest_ep, rewards

    def on_episode_end(self, episode, policy, rewards):
        if episode == 0 or episode % self.save_freq != 0:
            return

        print(f"[Callback] Saving checkpoint at episode {episode}")

        checkpoint = {
            "episode": episode,
            "policy": policy.state_dict(),
            "rewards": rewards,
        }

        path = os.path.join(self.save_dir, f"ppo_ep_{episode}.pt")
        torch.save(checkpoint, path)

        # Optional: save rewards separately for quick plotting
        np.save(os.path.join(self.save_dir, "rewards.npy"), rewards)
