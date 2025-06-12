import torch
import numpy as np
import oidn
from comfy.model_management import get_torch_device

class OIDN:
    def __init__(self):
        self.device = oidn.NewDevice()
        oidn.CommitDevice(self.device)
        self.filter = oidn.NewFilter(self.device, "RT")
        self.active = True

    def denoise(self, img_np):
        h, w, c = img_np.shape
        
        # Normalize if necessary
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        input_buf = img_np.astype(np.float32)
        output_buf = np.empty_like(input_buf)

        oidn.SetSharedFilterImage(self.filter, "color", input_buf, oidn.FORMAT_FLOAT3, w, h)
        oidn.SetSharedFilterImage(self.filter, "output", output_buf, oidn.FORMAT_FLOAT3, w, h)
        
        oidn.CommitFilter(self.filter)
        oidn.ExecuteFilter(self.filter)

        error_code = oidn.GetDeviceError(self.device)
        if error_code != 0:
            print(f"OIDN Error Code: {error_code}")
            return img_np # Return original on error
        
        return output_buf

    def __del__(self):
        if self.active:
            oidn.ReleaseFilter(self.filter)
            oidn.ReleaseDevice(self.device)
            self.active = False

class OIDNDenoiser:
    """
    A ComfyUI node to denoise images using Intel's Open Image Denoise (OIDN).
    This implementation is based on the best-practice script provided.
    """
    def __init__(self):
        self.torch_device = get_torch_device()
        self.oidn_instance = OIDN()

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the OIDN Denoiser node.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "Image Denoising"

    def denoise(self, image: torch.Tensor, denoise_strength: float):
        """
        Denoises the input image using OIDN.

        Args:
            image (torch.Tensor): The input image tensor (B, H, W, C).
            denoise_strength (float): The strength of the denoising.

        Returns:
            (torch.Tensor,): A tuple containing the denoised image tensor.
        """
        if denoise_strength == 0.0:
            return (image,)

        image_np = image.cpu().numpy()
        
        denoised_images = []
        for i in range(image_np.shape[0]):
            img = image_np[i]
            denoised_img = self.oidn_instance.denoise(img)
            if denoise_strength < 1.0:
                denoised_img = img * (1.0 - denoise_strength) + denoised_img * denoise_strength
            denoised_images.append(denoised_img)

        # Convert back to torch.Tensor
        denoised_tensor = torch.from_numpy(np.stack(denoised_images)).to(self.torch_device)
        
        return (denoised_tensor,)