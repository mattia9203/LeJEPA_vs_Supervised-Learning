import torch
import torch.nn.functional as F


class CNNGradCAM:
    def __init__(self, model, target_layer=None) -> None:
        self.model = model
        self.target_layer = target_layer or model.backbone.layer4
        self.activations = None
        self.gradients = None
        self.forward_handle = self.target_layer.register_forward_hook(
            self._capture_activations
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._capture_gradients
        )

    def _capture_activations(self, _module, _inputs, output):
        self.activations = output

    def _capture_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, images: torch.Tensor, class_indices=None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        if class_indices is None:
            class_indices = logits.argmax(dim=1)
        scores = logits.gather(1, class_indices[:, None]).sum()
        scores.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        maps = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        maps = F.interpolate(
            maps,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        maps = maps[:, 0]
        minimum = maps.flatten(1).min(dim=1).values[:, None, None]
        maximum = maps.flatten(1).max(dim=1).values[:, None, None]
        return (maps - minimum) / (maximum - minimum).clamp_min(1e-8)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()
