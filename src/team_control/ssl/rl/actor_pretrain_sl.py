import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from team_control.ssl.datasets.dataset import FootballDataset
from team_control.ssl.rl.actor_critic import Actor
from team_control.utils.visualization import plot_losses


# used as loss function for two-hot encoding of the regression
def soft_cross_entropy(pred, target):
    return -(target * torch.log(pred + 1e-8)).sum(dim=1).mean()


def pretrain_actor(epochs, sanity_check=False):
    device = torch.device("cpu")

    dataset = FootballDataset("../datasets/goalie.csv", device)

    if sanity_check:
        overfit_ds_size = 256
        dataset = torch.utils.data.Subset(
            dataset, list(range(overfit_ds_size)))

    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    actor = Actor().to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    train_losses = []
    actor.train()
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0

        for state, ball, teammates, opponents, action in train_loader:
            state = state.to(device)
            ball = ball.to(device)
            teammates = teammates.to(device)
            opponents = opponents.to(device)
            action = action.to(device)

            vx_t, vy_t, w_t, kick_t, dribble_t = action[:, 0], action[:, 1], action[:, 2], action[:, 3], action[:, 4]

            action_pred = actor(state, ball, teammates, opponents)
            vx_pred, vy_pred, w_pred, kick_pred, dribble_pred = action_pred

            loss = (
                soft_cross_entropy(vx_pred, vx_t) +
                soft_cross_entropy(vy_pred, vy_t) +
                soft_cross_entropy(w_pred, w_t) +
                soft_cross_entropy(kick_pred, kick_t) +
                soft_cross_entropy(dribble_pred, dribble_t)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(
            f"[Pretrain] Epoch {epoch:03d} | Loss {train_loss/len(train_loader):.4f}")
    plot_losses(train_losses=train_losses, val_losses=[])

    if sanity_check:
        # ---- Inspect predictions ----
        actor.eval()
        with torch.no_grad():
            state, ball, teammates, opponents, action = next(
                iter(train_loader))
            state = state.to(device)
            ball = ball.to(device)
            teammates = teammates.to(device)
            opponents = opponents.to(device)
            action = action.to(device)

            vx_t, vy_t, w_t, kick_t, dribble_t = action

            vx_pred, vy_pred, w_pred, kick_pred, dribble_pred = actor(
                state, ball, teammates, opponents)

            print("\nSample predictions vs ground truth:")
            for i in range(5):
                print(
                    f"GT:   {[vx_t[i], vy_t[i], w_t[i], kick_t[i], dribble_t[i]]}")
                print(
                    f"PRED: {[vx_pred[i], vy_pred[i], w_pred[i], kick_pred[i], dribble_pred[i]]}")
                print("-" * 80)

        print("=== Sanity Check Complete ===\n")

    else:
        # make it follow the same structure as the callback
        checkpoint = {
            'model_state_dict': actor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, "/Users/mahdigheidi/Documents/Univ/RoboCupLab/robocup-ssl/src/model_snapshots/actor_pretrained.pt")


if __name__ == "__main__":
    pretrain_actor(epochs=1000)
