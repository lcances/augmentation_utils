import torch
from typing import Callable


class Augmentation(torch.nn.Module):
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
    def __init__(self, pre_process_rule: Callable, post_process_rule: Callable, method='pick_one'):
        self.pre_process = []
        self.process = []
        self.post_process = []

        self.pre_process_rule = pre_process_rule
        self.post_process_rule = post_process_rule

        self.method = method

    def set_process(self, pool: list) -> None:
        self.process = pool

    def set_preprocess_rule(self, ruler: Callable) -> None:
        self.pre_process_rule = ruler
    
    def set_postprocess_rule(self, ruler: Callable) -> None:
        self.post_process_rule = ruler

    def compose(self, augmentation_pool: list) -> nn.Sequential:
        self.pre_process = []
        self.post_process = []

        if self.method == 'pick_one':
            return self._compose_pick_one(augmentation_pool)
        
        else:
            raise ValueError(f'Methods {self.method} doesn\'t exist.')

    def _compose_pick_one(self, augmentation_pool: list) -> nn.Sequential:
        """Select only one augmentation randomly."""
        aug_idx = random.randint(0, len(augmentation_pool) - 1)
        selected_aug = augmentation_pool[aug_idx]

        # check pre-process rule
        if self.pre_process_rule(selected_aug):
            self.pre_process = [selected_aug]

        elif self.post_process_rule(selected_aug):
            self.post_process = [selected_aug]

        # Compose the new sequential
        composed = nn.Sequential(
            nn.Sequential(*self.pre_process),
            self.process,
            nn.Sequential(*self.post_process),
        )

        return composed