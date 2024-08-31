import torch
import torch.distributed as dist
import time
import argparse
import os
import tempfile
import signal
import socket

# Set environment variables for NCCL to use Ethernet
os.environ["NCCL_SOCKET_IFNAME"] = "enp45s0f1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL's P2P functionality
os.environ["NCCL_SINGLE_RING_THRESHOLD"] = "0"

def initialize_process_group(backend, rank, world_size, master_addr, master_port):
    try:
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size
        )
    except Exception as e:
        print(f"Error initializing process group: {str(e)}")
        raise

def benchmark_transfer(tensor_size, num_iterations):
    # Initialize a random tensor
    tensor = torch.randn(tensor_size, device="cuda:0" if dist.get_rank() == 0 else "cuda:1")
    
    # Print the average of the tensor before sending
    avg_tensor = tensor.mean().item()
    print(f"[Rank {dist.get_rank()}] Average of tensor before sending: {avg_tensor:.4f}")
    
    # Warmup
    print(f"[Rank {dist.get_rank()}] Starting warmup...")
    for _ in range(5):
        if dist.get_rank() == 0:
            print(f"[Rank 0] Sending warmup tensor of size {tensor_size[0] / (1024 * 1024):.2f} MB to GPU 1")
            dist.send(tensor, dst=1)
        else:
            print(f"[Rank 1] Receiving warmup tensor of size {tensor_size[0] / (1024 * 1024):.2f} MB from GPU 0")
            dist.recv(tensor, src=0)
            print(f"[Rank 1] Received warmup tensor")
    
    torch.cuda.synchronize()
    print(f"[Rank {dist.get_rank()}] Warmup complete. Starting benchmark...")
    
    total_duration = 0.0
    for _ in range(num_iterations):
        start_time = time.time()
        if dist.get_rank() == 0:
            print(f"[Rank 0] Sending tensor to Rank 1, iteration {_ + 1}")
            dist.send(tensor, dst=1)
        else:
            print(f"[Rank 1] Receiving tensor from Rank 0, iteration {_ + 1}")
            dist.recv(tensor, src=0)
        torch.cuda.synchronize()  # Ensure the transfer is complete
        end_time = time.time()
        total_duration += (end_time - start_time)
    
    average_duration = total_duration / num_iterations
    total_bytes = tensor.nelement() * tensor.element_size() * num_iterations
    bandwidth = total_bytes / average_duration / 1e9  # GB/s
    
    # Print the average of the tensor after receiving
    if dist.get_rank() == 1:
        avg_received_tensor = tensor.mean().item()
        print(f"[Rank 1] Average of received tensor: {avg_received_tensor:.4f}")
    
    print(f"[Rank {dist.get_rank()}] Average Duration: {average_duration:.2f}s, Bandwidth: {bandwidth:.2f} GB/s")
    return bandwidth

def signal_handler(sig, frame):
    print("Signal received, cleaning up...")
    dist.destroy_process_group()
    exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Update the main function to include error handling
def main(args):
    try:
        initialize_process_group(args.backend, args.rank, args.world_size, args.master_addr, args.master_port)
        
        tensor_sizes = [
            (1 * 1024 * 1024 * 1024,),  # 1 GB
        ]
        
        print(f"[Rank {dist.get_rank()}] Running benchmark with {args.backend} backend")
        for size in tensor_sizes:
            bandwidth = benchmark_transfer(size, args.num_iterations)
            if dist.get_rank() == 0:
                print(f"[Rank 0] Tensor size: {size[0] / (1024 * 1024):.2f} MB, Bandwidth: {bandwidth:.2f} GB/s")
    except Exception as e:
        print(f"[Rank {dist.get_rank()}] Error in main: {str(e)}")
    finally:
        dist.destroy_process_group()
        print(f"[Rank {dist.get_rank()}] Process group destroyed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Distributed Benchmark")
    parser.add_argument("--backend", type=str, choices=["nccl"], required=True, help="Backend to use (nccl for GPU)")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process")
    parser.add_argument("--world-size", type=int, default=2, help="Total number of processes")
    parser.add_argument("--master-addr", type=str, required=True, help="Master node address")
    parser.add_argument("--master-port", type=str, required=True, help="Master node port")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Number of iterations for each tensor size")
    
    args = parser.parse_args()

    main(args)
