from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def get_model_name(self) -> str:
        pass

    def model_exists(self, name) -> bool:
        return False
