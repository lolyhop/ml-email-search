import torch
import time
import psutil
import tracemalloc

from src.retrievers.transformer import TransformerEmbedder

def measure_hardware_usage(model_name: str, model_id: str):
    tracemalloc.start()
    process = psutil.Process()

    start_ram = process.memory_info().rss / 1e9
    print(f"[{model_name}] RAM before load: {start_ram:.2f} GB")

    t0 = time.time()
    embedder = TransformerEmbedder(
        model_name=model_name,
        model_id=model_id,
        head="universal",
        head_size=384,
    )
    embedder.initialize()
    load_time = time.time() - t0

    current, peak = tracemalloc.get_traced_memory()
    end_ram = process.memory_info().rss / 1e9

    print(f"[{model_name}] Load time: {load_time:.2f} sec")
    print(f"[{model_name}] RAM after load: {end_ram:.2f} GB")
    print(f"[{model_name}] Peak RAM (tracemalloc): {peak / 1e6:.2f} MB")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{model_name}] Peak GPU memory: {gpu_mem:.2f} GB")

    tracemalloc.stop()


if __name__ == "__main__":
    measure_hardware_usage(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_id="hf://all-MiniLM-L6-v2"
    )
