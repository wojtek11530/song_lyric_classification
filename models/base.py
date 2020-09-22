from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    @abstractmethod
    def predict(self, lyric: str) -> np.ndarray:
        pass
