import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from team_control.ssl.rl.mappo import Actor
from team_control.utils.visualization import plot_losses


def pretrain_actor(epochs, sanity_check=False):
    device = torch.device("cpu")

    # read from the text file and create dataset
    # each line in the text is a comma separated list of floats
    # like this: -0.5786,0.7387,-0.5917,0.8062,-0.8126,-0.2113,-1.4771,0.7262,-0.9920,0.1518,-0.8447,1.5805,0.0000,0.0000,0.2416,0.6461,-1.5093,0.4091,-0.8900,-0.1149,-0.9588,1.4666,0.0000,0.0000,0.0000,0.0000,0.0000
    file = '../datasets/dataset.txt'
    observations, actions = [], []
    # we read line by line to avoid memory issues
    with open(file, 'defender.txt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = list(map(float, line.split(',')))
            # stack 10 of observations to form a single observation
            obs = []
            act = []
            observations.append(obs)
            actions.append(act)
    observations = torch.tensor(observations, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(observations, actions)


    if sanity_check:
        overfit_ds_size = 256
        dataset = torch.utils.data.Subset(dataset, list(range(overfit_ds_size)))

    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    actor = Actor().to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    train_losses = []
    actor.train()
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0

        # get 256 random samples from the dataset
        obs, action = next(iter(train_loader))
        # get the actions from the model one by one and compute loss
        obs = obs.to(device)
        action = action.to(device)

        for one_obs, one_action in zip(obs, action):
            one_obs = one_obs.unsqueeze(0)
            one_action = one_action.unsqueeze(0)
            action_pred, _, _ = actor(one_obs)
            vx_pred, vy_pred, w_pred, kick_pred, dribble_pred = action_pred

            loss = 0.0
            vx_t, vy_t, w_t, kick_t, dribble_t = one_action[0]

            loss = (
                torch.nn.functional.mse_loss(vx_pred, vx_t) +
                torch.nn.functional.mse_loss(vy_pred, vy_t) +
                torch.nn.functional.mse_loss(w_pred, w_t) +
                torch.nn.functional.binary_cross_entropy_with_logits(kick_pred, kick_t) +
                torch.nn.functional.mse_loss(dribble_pred, dribble_t)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()


        train_loss += loss.item()
        train_losses.append(train_loss)
        print(f"[Pretrain] Epoch {epoch:03d} | Loss {train_loss/len(train_loader):.4f}")
    plot_losses(train_losses=train_losses, val_losses=[])

    # if sanity_check:
    #     # ---- Inspect predictions ----
    #     actor.eval()
    #     with torch.no_grad():
    #         obs, action = next(iter(train_loader))
    #         obs = obs.to(device)
    #         action = action.to(device)

    #         vx_t, vy_t, w_t, kick_t, dribble_t = action[0]
    #         (vx_pred, vy_pred, w_pred, kick_pred, dribble_pred), _, _ = actor(obs)

    #         print("\nSample predictions vs ground truth:")
    #         for i in range(5):
    #             print(
    #                 f"GT:   {[vx_t[i], vy_t[i], w_t[i], kick_t[i], dribble_t[i]]}")
    #             print(
    #                 f"PRED: {[vx_pred[i], vy_pred[i], w_pred[i], kick_pred[i], dribble_pred[i]]}")
    #             print("-" * 80)

    print("=== Sanity Check Complete ===\n")

    # make it follow the same structure as the callback
    checkpoint = {
        'model_state_dict': actor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, "/Users/mahdigheidi/Documents/Univ/RoboCupLab/robocup-ssl/src/model_snapshots/actor_pretrained.pt")


if __name__ == "__main__":
    # Sanity check to overfit small dataset
    pretrain_actor(epochs=2000, sanity_check=True)
