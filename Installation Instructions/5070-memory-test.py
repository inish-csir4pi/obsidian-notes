#!/usr/bin/env python3
"""
RTX 5070 CUDA Out-of-Memory Test
Find the exact memory limits of your GPU
"""

import torch
import gc
import time

def print_memory_info():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB")

def test_single_tensor_limit():
    """Test maximum single tensor allocation"""
    print("="*60)
    print("SINGLE TENSOR MEMORY LIMIT TEST")
    print("="*60)
    
    device = torch.device('cuda')
    
    # Start with 100M elements and increase
    base_size = 100_000_000
    multiplier = 1.0
    max_successful = 0
    
    print("Testing maximum single tensor allocation...")
    print()
    
    while True:
        size = int(base_size * multiplier)
        memory_gb = size * 4 / (1024**3)  # float32 = 4 bytes
        
        print(f"Trying {size:,} elements ({memory_gb:.2f} GB)...")
        
        try:
            tensor = torch.randn(size, device=device, dtype=torch.float32)
            max_successful = size
            print(f"✅ SUCCESS!")
            print_memory_info()
            
            # Cleanup
            del tensor
            torch.cuda.empty_cache()
            
            # Increase by 10%
            multiplier *= 1.1
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ CUDA OUT OF MEMORY at {size:,} elements ({memory_gb:.2f} GB)")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            break
    
    print()
    print(f"🎯 MAXIMUM SINGLE TENSOR: {max_successful:,} elements ({max_successful * 4 / (1024**3):.2f} GB)")
    return max_successful

def test_multiple_tensor_limit():
    """Test how many tensors we can allocate before OOM"""
    print("\n" + "="*60)
    print("MULTIPLE TENSOR MEMORY LIMIT TEST")
    print("="*60)
    
    device = torch.device('cuda')
    tensors = []
    tensor_size = 50_000_000  # 50M elements per tensor
    tensor_gb = tensor_size * 4 / (1024**3)
    
    print(f"Allocating tensors of {tensor_size:,} elements ({tensor_gb:.2f} GB each)...")
    print()
    
    try:
        count = 0
        while True:
            count += 1
            print(f"Allocating tensor #{count}...")
            
            tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)
            tensors.append(tensor)
            
            total_allocated = len(tensors) * tensor_gb
            print(f"✅ Success! Total: {total_allocated:.2f} GB")
            print_memory_info()
            print()
            
    except torch.cuda.OutOfMemoryError:
        print(f"❌ CUDA OUT OF MEMORY after {len(tensors)} tensors")
        print(f"🎯 MAXIMUM TENSORS: {len(tensors)} × {tensor_gb:.2f} GB = {len(tensors) * tensor_gb:.2f} GB")
        torch.cuda.empty_cache()
        return len(tensors)
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return len(tensors)

def test_model_memory_limit():
    """Test maximum model size before OOM"""
    print("\n" + "="*60)
    print("MODEL MEMORY LIMIT TEST")  
    print("="*60)
    
    device = torch.device('cuda')
    
    # Test different model sizes
    layer_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    for hidden_size in layer_sizes:
        print(f"\nTesting model with hidden size: {hidden_size}")
        
        try:
            # Create large model
            model = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1000)
            ).to(device)
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            param_gb = param_count * 4 / (1024**3)  # float32
            
            print(f"✅ Model created successfully!")
            print(f"   Parameters: {param_count:,} ({param_gb:.2f} GB)")
            print_memory_info()
            
            # Test forward pass with large batch
            batch_size = 64
            try:
                data = torch.randn(batch_size, hidden_size, device=device)
                output = model(data)
                print(f"✅ Forward pass successful with batch size {batch_size}")
                print_memory_info()
                
                # Test backward pass
                loss = output.sum()
                loss.backward()
                print(f"✅ Backward pass successful!")
                print_memory_info()
                
            except torch.cuda.OutOfMemoryError:
                print(f"❌ Forward/backward pass failed - OOM during training")
            
            # Cleanup
            del model, data, output, loss
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ CUDA OUT OF MEMORY - Cannot create model with hidden size {hidden_size}")
            print(f"🎯 MAXIMUM MODEL SIZE: Previous size was the limit")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            break

def test_batch_size_limit():
    """Test maximum batch size for a fixed model"""
    print("\n" + "="*60)
    print("BATCH SIZE LIMIT TEST")
    print("="*60)
    
    device = torch.device('cuda')
    
    # Fixed model size
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096), 
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 1000)
    ).to(device)
    
    print("Testing maximum batch size with 2048->4096->4096->1000 model...")
    print()
    
    batch_size = 1
    max_batch = 0
    
    while True:
        print(f"Testing batch size: {batch_size}")
        
        try:
            # Forward pass
            data = torch.randn(batch_size, 2048, device=device)
            output = model(data)
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            max_batch = batch_size
            print(f"✅ Success!")
            print_memory_info()
            
            # Cleanup gradients
            model.zero_grad()
            del data, output, loss
            torch.cuda.empty_cache()
            
            # Increase batch size
            if batch_size < 10:
                batch_size += 1
            elif batch_size < 100:
                batch_size += 10
            else:
                batch_size += 50
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ CUDA OUT OF MEMORY at batch size {batch_size}")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            break
    
    print(f"\n🎯 MAXIMUM BATCH SIZE: {max_batch}")

def stress_test_memory():
    """Continuous memory allocation/deallocation stress test"""
    print("\n" + "="*60)
    print("MEMORY STRESS TEST")
    print("="*60)
    
    device = torch.device('cuda')
    
    print("Running continuous allocation/deallocation for 60 seconds...")
    print("Press Ctrl+C to stop early")
    print()
    
    start_time = time.time()
    iterations = 0
    
    try:
        while time.time() - start_time < 60:  # Run for 60 seconds
            # Allocate large tensor
            tensor = torch.randn(100_000_000, device=device)  # ~400MB
            
            # Do some computation
            result = tensor * 2 + 1
            
            # Deallocate
            del tensor, result
            torch.cuda.empty_cache()
            
            iterations += 1
            
            if iterations % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   Iteration {iterations}, Elapsed: {elapsed:.1f}s")
                print_memory_info()
                
    except torch.cuda.OutOfMemoryError:
        print(f"❌ CUDA OUT OF MEMORY during stress test at iteration {iterations}")
    except KeyboardInterrupt:
        print(f"\n⚠️  Test stopped by user after {iterations} iterations")
    
    elapsed = time.time() - start_time
    print(f"\n🎯 STRESS TEST COMPLETE: {iterations} iterations in {elapsed:.1f}s")
    print(f"   Allocations per second: {iterations/elapsed:.2f}")

def main():
    """Run all memory limit tests"""
    print("RTX 5070 CUDA OUT-OF-MEMORY LIMIT FINDER")
    print("This will find the exact memory limits of your GPU")
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
    print()
    
    # Clear memory first
    torch.cuda.empty_cache()
    
    try:
        # Run tests
        test_single_tensor_limit()
        test_multiple_tensor_limit() 
        test_model_memory_limit()
        test_batch_size_limit()
        stress_test_memory()
        
        print("\n" + "="*60)
        print("🎉 ALL MEMORY TESTS COMPLETE!")
        print("="*60)
        print("You now know the exact memory limits of your RTX 5070!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()