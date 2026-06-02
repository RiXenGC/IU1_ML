import torch
from torch.utils.data import DataLoader
import tqdm
import copy


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> None:
        self.model: torch.nn.Module = model
        self.criterion: torch.nn.Module = criterion
        self.optimizer: torch.optim.Optimizer = optimizer
        self.device: str = device

    def _process_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def train_epoch(
        self, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in tqdm.tqdm(dataloader, desc="Training", leave=False):
            inputs, targets = self._process_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

        return running_loss / total_samples if total_samples > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            inputs, targets = self._process_batch(batch)

            outputs: torch.Tensor = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss = running_loss + loss.item() * targets.size(0)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 3,
        min_delta: float = 0.001,
    ) -> dict[str, list[float]]:
        best_loss = float("inf")
        patience_counter = 0

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            if val_loss < (best_loss - min_delta):
                best_loss = val_loss
                patience_counter = 0
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                print("Saved best model")
            else:
                patience_counter += 1
                print(f"No Improvements: {patience_counter}/{patience}")

            # Триггер ранней остановки
            if patience_counter >= patience:
                print(f"Early stopping: {epoch}")
                break

        self.model.load_state_dict(self.best_model_wts)
        print("Successfully loaded best model weights.")
        return history
