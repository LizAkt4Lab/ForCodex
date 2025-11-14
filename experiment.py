import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency for plotting
    plt = None
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency for plotting utilities
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PyTorch is required for this experiment. Please install torch (CPU is sufficient)."
    ) from exc


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class ModelSettings:
    num_users: int
    num_times: int
    num_topics: int
    vocab_size: int
    dx: int
    du: int
    clicks_per_time: Tuple[int, int]


class UserSequenceDataset(Dataset):
    def __init__(self, users: List[Dict]):
        self.users = users

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict:
        return self.users[idx]


class AlphaEncoder(nn.Module):
    def __init__(self, vocab_size: int, topic_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    def forward(self, bow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(bow))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class SPosteriorNet(nn.Module):
    def __init__(self, dx: int, du: int, topic_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(dx + du + topic_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    def forward(
        self,
        user_x: torch.Tensor,
        control: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([user_x, control, prev_state], dim=-1)
        h = F.relu(self.fc1(features))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class KappaNet(nn.Module):
    def __init__(self, dx: int, topic_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dx, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, topic_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicTopicModel(nn.Module):
    def __init__(self, settings: ModelSettings):
        super().__init__()
        self.settings = settings
        K = settings.num_topics
        V = settings.vocab_size
        dx = settings.dx
        du = settings.du

        self.kappa_net = KappaNet(dx, K)
        self.alpha_encoder = AlphaEncoder(V, K)
        self.s_posterior = SPosteriorNet(dx, du, K)

        self.A = nn.Parameter(torch.eye(K) * 0.5 + 0.05 * torch.randn(K, K))
        self.B = nn.Parameter(0.05 * torch.randn(K, du))
        self.log_diag_Q = nn.Parameter(torch.full((K,), math.log(0.1)))
        self.W_theta = nn.Parameter(torch.randn(K, K) * 0.1)
        self.beta = nn.Parameter(torch.randn(K, V) * 0.1)

        self.lambda_kappa = 1e-3

    def kl_standard_normal(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def kl_state(self, mean: torch.Tensor, logvar: torch.Tensor, prior_mean: torch.Tensor) -> torch.Tensor:
        var = logvar.exp()
        diag_Q = self.log_diag_Q.exp()
        kl = 0.5 * torch.sum(
            (var + (mean - prior_mean).pow(2)) / diag_Q
            + self.log_diag_Q
            - logvar
            - 1
        )
        return kl

    def forward_user(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        K = self.settings.num_topics
        user_x = batch["x"].to(device)
        kappa = self.kappa_net(user_x)
        kappa_true = batch.get("kappa_true")
        if kappa_true is not None:
            kappa_penalty = self.lambda_kappa * torch.sum((kappa - kappa_true.to(device)).pow(2))
        else:
            kappa_penalty = self.lambda_kappa * torch.sum(kappa.pow(2))

        recon_loss = torch.tensor(0.0, device=device)
        kl_alpha = torch.tensor(0.0, device=device)
        kl_s = torch.tensor(0.0, device=device)

        prev_state = torch.zeros(K, device=device)
        for time_step in batch["clicks"]:
            u_t = time_step["u"].to(device)
            mean_s, logvar_s = self.s_posterior(user_x, u_t, prev_state)
            eps_s = torch.randn_like(mean_s)
            s_sample = mean_s + eps_s * torch.exp(0.5 * logvar_s)

            prior_mean = torch.matmul(self.A, prev_state) + torch.matmul(self.B, u_t)
            kl_s = kl_s + self.kl_state(mean_s, logvar_s, prior_mean)

            eta = F.softmax(kappa + s_sample, dim=-1)

            for doc in time_step["docs"]:
                bow = doc["bow"].to(device)
                bow_norm = bow / (bow.sum() + 1e-8)
                mu_alpha, logvar_alpha = self.alpha_encoder(bow_norm)
                eps_alpha = torch.randn_like(mu_alpha)
                alpha_sample = mu_alpha + eps_alpha * torch.exp(0.5 * logvar_alpha)

                kl_alpha = kl_alpha + self.kl_standard_normal(mu_alpha, logvar_alpha)

                interaction = alpha_sample * eta
                theta_logits = torch.matmul(self.W_theta, interaction)
                theta = F.softmax(theta_logits, dim=-1)

                word_logits = torch.matmul(theta, self.beta)
                log_word_probs = F.log_softmax(word_logits, dim=-1)
                recon_loss = recon_loss - torch.sum(bow * log_word_probs)

            prev_state = s_sample

        total_loss = recon_loss + kl_alpha + kl_s + kappa_penalty
        return {
            "loss": total_loss,
            "recon": recon_loss.detach(),
            "kl_alpha": kl_alpha.detach(),
            "kl_s": kl_s.detach(),
        }

    @torch.no_grad()
    def infer_user(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        K = self.settings.num_topics
        user_x = batch["x"].to(device)
        kappa = self.kappa_net(user_x)
        prev_state = torch.zeros(K, device=device)
        means = []
        etas = []
        for time_step in batch["clicks"]:
            u_t = time_step["u"].to(device)
            mean_s, logvar_s = self.s_posterior(user_x, u_t, prev_state)
            means.append(mean_s.cpu())
            eta = F.softmax(kappa + mean_s, dim=-1)
            etas.append(eta.cpu())
            prev_state = mean_s
        return {"s_means": torch.stack(means), "etas": torch.stack(etas)}


def generate_synthetic_data(settings: ModelSettings, seed: int = 0) -> Dict:
    set_seed(seed)
    K = settings.num_topics
    V = settings.vocab_size
    N = settings.num_users
    T = settings.num_times
    dx = settings.dx
    du = settings.du

    x_users = torch.randn(N, dx)
    u_controls = 0.5 * torch.randn(T, du)

    W_kappa = 0.3 * torch.randn(K, dx)
    b_kappa = 0.1 * torch.randn(K)

    A_true = 0.6 * torch.eye(K) + 0.05 * torch.randn(K, K)
    eigvals = torch.linalg.eigvals(A_true).abs()
    spectral_radius = eigvals.max().item()
    if spectral_radius >= 1.0:
        A_true = A_true / (spectral_radius + 0.1)

    B_true = 0.2 * torch.randn(K, du)
    Q_diag = 0.05 + 0.02 * torch.rand(K)
    W_theta_true = 0.3 * torch.randn(K, K)

    gamma_dist = torch.distributions.Gamma(torch.tensor(1.0), torch.tensor(1.0))
    beta_true = gamma_dist.sample((K, V))
    beta_true = beta_true / beta_true.sum(dim=1, keepdim=True)

    users = []
    doc_id = 0
    all_docs = {}

    for i in range(N):
        x_i = x_users[i]
        kappa_i = torch.matmul(W_kappa, x_i) + b_kappa
        s_prev = torch.zeros(K)
        user_clicks = []
        s_seq = []
        eta_seq = []

        for t in range(T):
            noise = torch.sqrt(Q_diag) * torch.randn(K)
            s_t = torch.matmul(A_true, s_prev) + torch.matmul(B_true, u_controls[t]) + noise
            eta_t = torch.softmax(kappa_i + s_t, dim=0)

            time_docs = []
            num_clicks = random.randint(settings.clicks_per_time[0], settings.clicks_per_time[1])
            for _ in range(num_clicks):
                alpha_d = torch.randn(K)
                interaction = alpha_d * eta_t
                theta_logits = torch.matmul(W_theta_true, interaction)
                theta = torch.softmax(theta_logits, dim=0)
                word_logits = torch.matmul(beta_true.t(), theta)
                word_probs = torch.softmax(word_logits, dim=0)
                doc_length = int(torch.poisson(torch.tensor(80.0)).item()) + 5
                word_indices = torch.multinomial(word_probs, doc_length, replacement=True)
                counts = torch.bincount(word_indices, minlength=V).to(torch.float32)
                bow = counts

                time_docs.append(
                    {
                        "doc_id": doc_id,
                        "bow": bow,
                        "alpha_true": alpha_d,
                        "theta_true": theta,
                    }
                )
                all_docs[doc_id] = {
                    "alpha_true": alpha_d,
                    "bow": bow,
                }
                doc_id += 1

            user_clicks.append(
                {
                    "time": t,
                    "u": u_controls[t],
                    "docs": time_docs,
                }
            )
            s_seq.append(s_t)
            eta_seq.append(eta_t)
            s_prev = s_t

        users.append(
            {
                "user_id": i,
                "x": x_i,
                "clicks": user_clicks,
                "s_true": torch.stack(s_seq),
                "eta_true": torch.stack(eta_seq),
                "kappa_true": kappa_i,
            }
        )

    data = {
        "users": users,
        "docs": all_docs,
        "settings": settings,
        "params": {
            "A_true": A_true,
            "B_true": B_true,
            "Q_true": Q_diag,
            "beta_true": beta_true,
            "W_theta_true": W_theta_true,
            "W_kappa": W_kappa,
            "b_kappa": b_kappa,
        },
        "controls": u_controls,
    }
    return data


def train_model(data: Dict, device: torch.device, epochs: int = 30) -> Tuple[DynamicTopicModel, List[Dict[str, float]]]:
    settings = data["settings"]
    model = DynamicTopicModel(settings).to(device)

    dataset = UserSequenceDataset(data["users"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl_alpha = 0.0
        epoch_kl_s = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model.forward_user(batch, device)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += outputs["loss"].item()
            epoch_recon += outputs["recon"].item()
            epoch_kl_alpha += outputs["kl_alpha"].item()
            epoch_kl_s += outputs["kl_s"].item()

        avg_loss = epoch_loss / len(dataset)
        avg_recon = epoch_recon / len(dataset)
        avg_kl_alpha = epoch_kl_alpha / len(dataset)
        avg_kl_s = epoch_kl_s / len(dataset)
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "recon": avg_recon,
                "kl_alpha": avg_kl_alpha,
                "kl_s": avg_kl_s,
            }
        )
        print(
            f"Epoch {epoch:02d} | Loss: {avg_loss:.2f} | Recon: {avg_recon:.2f} | KL_alpha: {avg_kl_alpha:.2f} | KL_s: {avg_kl_s:.2f}"
        )

    return model, history


def analyze_results(model: DynamicTopicModel, data: Dict, device: torch.device, output_dir: Path) -> None:
    if plt is None or np is None:
        raise RuntimeError(
            "Matplotlib and NumPy are required for analysis but are not installed in this environment."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    inferred = {}
    for user in data["users"]:
        inferred[user["user_id"]] = model.infer_user(user, device)

    A_true = data["params"]["A_true"].numpy()
    B_true = data["params"]["B_true"].numpy()
    A_est = model.A.detach().cpu().numpy()
    B_est = model.B.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    im0 = axes[0, 0].imshow(A_true, cmap="coolwarm")
    axes[0, 0].set_title("A True")
    fig.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(A_est, cmap="coolwarm")
    axes[0, 1].set_title("A Estimated")
    fig.colorbar(im1, ax=axes[0, 1])
    im2 = axes[1, 0].imshow(B_true, cmap="coolwarm")
    axes[1, 0].set_title("B True")
    fig.colorbar(im2, ax=axes[1, 0])
    im3 = axes[1, 1].imshow(B_est, cmap="coolwarm")
    axes[1, 1].set_title("B Estimated")
    fig.colorbar(im3, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "ab_heatmaps.png", dpi=200)

    frob_A = np.linalg.norm(A_est - A_true)
    frob_B = np.linalg.norm(B_est - B_true)
    print(f"Frobenius error |A - A_true| = {frob_A:.4f}")
    print(f"Frobenius error |B - B_true| = {frob_B:.4f}")

    chosen_users = random.sample(list(inferred.keys()), k=3)
    time_axis = np.arange(data["settings"].num_times)
    for user_id in chosen_users:
        s_true = data["users"][user_id]["s_true"].numpy()
        s_mean = inferred[user_id]["s_means"].numpy()
        fig, axes = plt.subplots(model.settings.num_topics, 1, figsize=(10, 2 * model.settings.num_topics))
        if model.settings.num_topics == 1:
            axes = [axes]
        for k in range(model.settings.num_topics):
            axes[k].plot(time_axis, s_true[:, k], label="True", color="C0")
            axes[k].plot(time_axis, s_mean[:, k], label="Estimated", color="C1", linestyle="--")
            axes[k].set_ylabel(f"s_dim {k}")
        axes[0].set_title(f"User {user_id} state trajectories")
        axes[-1].set_xlabel("Time")
        axes[0].legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"user_{user_id}_states.png", dpi=200)

    beta_true = data["params"]["beta_true"].numpy()
    beta_est = F.softmax(model.beta, dim=-1).detach().cpu().numpy()
    vocab = [f"w{i}" for i in range(data["settings"].vocab_size)]

    cosine_sims = []
    top_n = 10
    for k in range(model.settings.num_topics):
        true_probs = beta_true[k]
        est_probs = beta_est[k]
        top_true_idx = np.argsort(true_probs)[-top_n:][::-1]
        top_est_idx = np.argsort(est_probs)[-top_n:][::-1]
        print(f"Topic {k} top-{top_n} words (true vs est):")
        print(" True:", [vocab[idx] for idx in top_true_idx])
        print(" Est:", [vocab[idx] for idx in top_est_idx])

        cosine = np.dot(true_probs, est_probs) / (
            np.linalg.norm(true_probs) * np.linalg.norm(est_probs) + 1e-8
        )
        cosine_sims.append(cosine)

        fig = plt.figure(figsize=(10, 4))
        indices = np.arange(top_n)
        plt.bar(indices - 0.2, true_probs[top_true_idx], width=0.4, label="True")
        plt.bar(indices + 0.2, est_probs[top_true_idx], width=0.4, label="Est (true idx)")
        plt.xticks(indices, [vocab[idx] for idx in top_true_idx], rotation=45)
        plt.title(f"Topic {k} word probs (top-{top_n} true indices)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"topic_{k}_words.png", dpi=200)

    avg_cosine = float(np.mean(cosine_sims))
    print(f"Average cosine similarity between beta topics: {avg_cosine:.4f}")

    user_id = random.choice(list(inferred.keys()))
    topic_dim = random.randrange(model.settings.num_topics)
    eta_true = data["users"][user_id]["eta_true"].numpy()[:, topic_dim]
    eta_est = inferred[user_id]["etas"].numpy()[:, topic_dim]

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, eta_true, label="True eta")
    plt.plot(time_axis, eta_est, label="Estimated eta", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel(f"Eta dim {topic_dim}")
    plt.title(f"User {user_id} eta comparison (topic {topic_dim})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eta_comparison.png", dpi=200)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic topic model experiment")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (e.g. cpu or cuda)")
    parser.add_argument("--num-users", type=int, default=50, help="Number of users")
    parser.add_argument("--num-times", type=int, default=20, help="Number of time steps")
    parser.add_argument("--num-topics", type=int, default=5, help="Number of latent topics")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--dx", type=int, default=8, help="User feature dimension")
    parser.add_argument("--du", type=int, default=3, help="Control feature dimension")
    parser.add_argument(
        "--clicks-range",
        type=int,
        nargs=2,
        default=(1, 3),
        metavar=("MIN", "MAX"),
        help="Inclusive range for number of clicks per user/time step",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to store analysis figures",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis plots even if matplotlib/numpy are available",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    settings = ModelSettings(
        num_users=args.num_users,
        num_times=args.num_times,
        num_topics=args.num_topics,
        vocab_size=args.vocab_size,
        dx=args.dx,
        du=args.du,
        clicks_per_time=(args.clicks_range[0], args.clicks_range[1]),
    )

    data = generate_synthetic_data(settings, seed=args.seed)
    device = torch.device(args.device)
    model, history = train_model(data, device=device, epochs=args.epochs)

    print("Training complete. Final epoch stats:")
    if history:
        final = history[-1]
        print(
            f"Epoch {final['epoch']:02d}: loss={final['loss']:.4f}, recon={final['recon']:.4f}, "
            f"kl_alpha={final['kl_alpha']:.4f}, kl_s={final['kl_s']:.4f}"
        )

    if args.skip_analysis or plt is None:
        if plt is None and not args.skip_analysis:
            print("Matplotlib not available; skipping analysis plots.")
        return

    analyze_results(model, data, device=device, output_dir=args.analysis_dir)


if __name__ == "__main__":
    main()
