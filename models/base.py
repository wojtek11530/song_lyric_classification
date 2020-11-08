from abc import abstractmethod
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule):
    @abstractmethod
    def predict(self, lyrics: str) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def test_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.eval()
        predictions: List[torch.Tensor] = []
        real_values: List[torch.Tensor] = []
        with torch.no_grad():
            dataloader = model.test_dataloader()
            total = len(dataloader.dataset)
            current = 0
            for batch in dataloader:
                y_labels, y_hat = self._batch_step(batch)
                predictions.extend(y_hat)
                real_values.extend(y_labels)
                current += len(y_hat)
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: {current}/{total}")

        predictions_tensor = torch.stack(predictions).cpu()
        real_values_tensor = torch.stack(real_values).cpu()
        return predictions_tensor, real_values_tensor

    @abstractmethod
    def _batch_step(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
