import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, obj): # [B, input_dim] -> [B, 64]
        return self.net(obj)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, ego_emb, other_embs):
        """
        ego_emb:     Tensor [B, 64]
        other_embs:  Tensor [B, N, 64]

        Returns:
            pooled embedding: [D]
        """
        q = self.query(ego_emb).unsqueeze(1)      # [B, 1, D]
        k = self.key(other_embs)                  # [B, N, D]
        v = self.value(other_embs)                # [B, N, D]

        k_t = k.transpose(-2, -1)
        scores = (q @ k_t) * self.scale           # [B, 1, N]
        weights = F.softmax(scores, dim=-1)       # [B, 1, N]

        pooled = weights @ v                      # [B, 1, D]
        return pooled.squeeze(1)                  # [B, D]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.robot_encoder = ObjectEncoder(input_dim=6)  # 6 -> 64
        self.ball_encoder = ObjectEncoder(input_dim=2)  # 2 -> 64

        self.pooling = AttentionPooling(embed_dim=64)

    def forward(self, robot, ball, team_robots, opp_robots):
        robot_emb = self.robot_encoder(robot)  # [B, 64]
        ball_emb = self.ball_encoder(ball)  # [B, 64]
        team_emb = self.robot_encoder(team_robots)  # [B, N_t, 64]
        opp_emb = self.robot_encoder(opp_robots)  # [B, N_o, 64]

        team_pooled = self.pooling(robot_emb, team_emb)
        opp_pooled = self.pooling(robot_emb, opp_emb)

        return torch.cat([robot_emb, team_pooled, opp_pooled, ball_emb], dim=-1)


if __name__ == '__main__':
    torch.manual_seed(0)

    encoder = Encoder()
    encoder.eval()

    def run_test(name, robot, ball, team, opp):
        print(f"Running test: {name}")
        out = encoder(robot, ball, team, opp)

        # Shape checks
        assert out.ndim == 2, f"{name}: Output must be 2D [B, 256], got {out.shape}"
        assert out.shape[0] == robot.shape[0], f"{name}: Batch size mismatch"
        assert out.shape[1] == 256, f"{name}: Embedding dim must be 256, got {out.shape[1]}"

        # Finite value checks
        assert torch.isfinite(out).all(), f"{name}: Output contains NaNs or Infs"

        print(f"  ✓ shape: {out.shape}")
        print(f"  ✓ finite values\n")

    # -----------------------
    # Test 1: Single batch, multiple teammates and opponents
    # -----------------------
    B = 1
    robot = torch.randn(B, 6)
    ball = torch.randn(B, 2)
    team = torch.randn(B, 3, 6)
    opp = torch.randn(B, 2, 6)
    run_test("Single batch, N_t=3, N_o=2", robot, ball, team, opp)

    # -----------------------
    # Test 2: Batch size > 1, varying team sizes
    # -----------------------
    B = 4
    robot = torch.randn(B, 6)
    ball = torch.randn(B, 2)
    team = torch.randn(B, 5, 6)
    opp = torch.randn(B, 1, 6)
    run_test("Batch=4, N_t=5, N_o=1", robot, ball, team, opp)

    # -----------------------
    # Test 3: No teammates
    # -----------------------
    B = 2
    robot = torch.randn(B, 6)
    ball = torch.randn(B, 2)
    team = torch.zeros(B, 1, 6)
    opp = torch.randn(B, 4, 6)
    run_test("No teammates", robot, ball, team, opp)

    # -----------------------
    # Test 4: No opponents
    # -----------------------
    B = 3
    robot = torch.randn(B, 6)
    ball = torch.randn(B, 2)
    team = torch.randn(B, 2, 6)
    opp = torch.zeros(B, 1, 6)
    run_test("No opponents", robot, ball, team, opp)

    # -----------------------
    # Test 5: No teammates and no opponents
    # -----------------------
    B = 5
    robot = torch.randn(B, 6)
    ball = torch.randn(B, 2)
    team = torch.zeros(B, 1, 6)
    opp = torch.zeros(B, 1, 6)
    run_test("No teammates and no opponents", robot, ball, team, opp)

    # -----------------------
    # Test 6: Gradient flow sanity check
    # -----------------------
    print("Running test: Gradient flow check")
    robot = torch.randn(2, 6, requires_grad=True)
    ball = torch.randn(2, 2, requires_grad=True)
    team = torch.randn(2, 3, 6, requires_grad=True)
    opp = torch.randn(2, 2, 6, requires_grad=True)

    encoder.train()
    out = encoder(robot, ball, team, opp)
    loss = out.mean()
    loss.backward()

    grad_ok = True
    for name, param in encoder.named_parameters():
        if param.grad is None:
            print(f"  ⚠ No gradient for {name}")
            grad_ok = False

    assert grad_ok, "Some encoder parameters did not receive gradients"
    print("  ✓ All parameters received gradients")

    print("\nAll encoder tests passed successfully.")
