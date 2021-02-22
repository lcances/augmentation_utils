from torch.nn import Module, Sequential
from typing import Callable
import random


class Augmentation(Module):
    def __init__(self, ratio: float):
        super().__init__()
        assert 0.0 <= ratio <= 1.0

        self.ratio = ratio

    def forward(self, x):
        return x

    def __copy__(self):
        newone = type(self)(self.ratio)
        newone.__dict__.update(self.__dict__)
        return newone


class SignalAugmentation(Augmentation):
    def __init__(self, ratio):
        super().__init__(ratio)


class SpecAugmentation(Augmentation):
    def __init__(self, ratio):
        super().__init__(ratio)


class ImgAugmentation(Augmentation):
    def __init__(self, ratio):
        super().__init__(ratio)


class ComposeAugmentation:
    def __init__(self, pre_process_rule: Callable, post_process_rule: Callable, method='pick-one'):
        super().__init__()
        self.augmentation_pool = []
        self.pre_process = []
        self.process = []
        self.post_process = []

        self.pre_process_rule = pre_process_rule
        self.post_process_rule = post_process_rule

        self.method = method

    def set_process(self, pool: list) -> None:
        self.process = pool

    def set_augmentation_pool(self, pool: list) -> None:
        self.augmentation_pool = pool

    def __call__(self, x) -> Sequential:
        self.pre_process = []
        self.post_process = []

        if self.method == 'pick-one':
            return self._compose_pick_one()(x)

        else:
            raise ValueError(f'Methods {self.method} doesn\'t exist.')

    def _compose_pick_one(self) -> Sequential:
        """Select only one augmentation randomly."""
        aug_idx = random.randint(0, len(self.augmentation_pool) - 1)
        selected_aug = self.augmentation_pool[aug_idx]

        # check pre-process rule
        if self.pre_process_rule(selected_aug):
            self.pre_process = [selected_aug]

        elif self.post_process_rule(selected_aug):
            self.post_process = [selected_aug]

        # Compose the new sequential
        composed = Sequential(
            Sequential(*self.pre_process),
            self.process,
            Sequential(*self.post_process),
        )

        return composed