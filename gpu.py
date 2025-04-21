import torch

def check_gpu_availability():
    """Checks if CUDA is available and provides information about the GPU."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA is available! Found {gpu_count} GPU(s).")
        for i in range(gpu_count):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Current device index: {torch.cuda.current_device()}")
        print(f"  - GPU properties (device 0): {torch.cuda.get_device_properties(0) if gpu_count > 0 else 'N/A'}")
        print(f"  - CUDA version (linked with PyTorch): {torch.version.cuda}")
        return True
    else:
        print("CUDA is not available. Training will be performed on the CPU.")
        return False

if __name__ == "__main__":
    gpu_ready = check_gpu_availability()
    print(f"\nGPU Ready for Use: {gpu_ready}")

    # Example of moving a tensor to the GPU if available
    if gpu_ready:
        device = torch.device("cuda")
        sample_tensor = torch.randn(3, 3).to(device)
        print(f"\nSample tensor on GPU:\n{sample_tensor}")
        print(f"Tensor device: {sample_tensor.device}")
    else:
        device = torch.device("cpu")
        sample_tensor = torch.randn(3, 3).to(device)
        print(f"\nSample tensor on CPU:\n{sample_tensor}")
        print(f"Tensor device: {sample_tensor.device}")