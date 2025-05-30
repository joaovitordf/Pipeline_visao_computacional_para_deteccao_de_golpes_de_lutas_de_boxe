"""
pip uninstall torch torchvision torchaudio -

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

"""
Esse script serve pra saber se tem GPU com suporte cuda no pc
"""

import torch

print("PyTorch versão:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
else:
    print("Nenhuma GPU detectada.")
