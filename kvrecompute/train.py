# SPDX-License-Identifier: Apache-2.0

"""Training entry point for token-level recompute score models."""

# Standard
import argparse
from pathlib import Path

# Third Party
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# First Party
from kvrecompute.data import FEATURE_NAMES, TokenScoreDataset, load_teacher_dataset
from kvrecompute.model import TokenMLPRegressor


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a token-level recompute scorer.")
    parser.add_argument("--dataset", required=True, help="Path to a precomputed teacher dataset (.pt).")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="MLP dropout.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Token batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one train or eval epoch and return average loss."""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        predictions = model(features)
        loss = criterion(predictions, labels)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_count = features.shape[0]
        total_loss += loss.item() * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


def main() -> None:
    """Train the recompute MLP on a precomputed token-level teacher dataset."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_teacher_dataset(args.dataset)
    dataset = TokenScoreDataset(tensors)

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TokenMLPRegressor(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=args.device,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=args.device,
            optimizer=None,
        )
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_names": FEATURE_NAMES,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_val_loss": best_val_loss,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    torch.save(best_state, output_dir / "token_mlp.pt")
    print(f"saved checkpoint to {output_dir / 'token_mlp.pt'}")


if __name__ == "__main__":
    main()

