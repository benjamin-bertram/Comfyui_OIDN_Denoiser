from .oidn_denoiser import OIDNDenoiser

NODE_CLASS_MAPPINGS = {
    "OIDNDenoiser": OIDNDenoiser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OIDNDenoiser": "OIDN Denoiser"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']