#!/usr/bin/env python3
"""
PyTorch GPU Detection and Performance Test for RTX 5070
This script checks if PyTorch can detect and utilize your RTX 5070 GPU
"""

import torch
import time
import sys

def check_pytorch_installation():
    """Check basic PyTorch installation and CUDA availability"""
    print("="*60)
    print("PyTorch Installation Check")
    print("="*60)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("❌ CUDA is not available. PyTorch will run on CPU only.")
        return False
    
    return True

def check_gpu_details():
    """Check detailed GPU information"""
    print("\n" + "="*60)
    print("GPU Hardware Information")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("No CUDA GPUs detected.")
        return False
    
    rtx_5070_found = False
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        
        print(f"GPU {i}: {gpu_name}")
        print(f"  Memory: {gpu_memory:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Check if RTX 5070 is detected
        if "5070" in gpu_name or "RTX 5070" in gpu_name.upper():
            rtx_5070_found = True
            print(f"  ✅ RTX 5070 detected!")
        
        print()
    
    if not rtx_5070_found:
        print("⚠️  RTX 5070 not specifically detected. Available GPU(s) listed above.")
    
    return rtx_5070_found

def test_gpu_computation():
    """Test actual GPU computation with PyTorch"""
    print("="*60)
    print("GPU Computation Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Cannot test GPU computation - CUDA not available.")
        return False
    
    # Set device
    device = torch.device('cuda:0')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Test tensor operations on GPU
    try:
        print("\n1. Testing basic tensor operations...")
        
        # Create tensors on GPU
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        gpu_time = time.time() - start_time
        
        print(f"   Matrix multiplication (1000x1000): {gpu_time:.4f} seconds")
        print(f"   Result tensor shape: {c.shape}")
        print(f"   Result tensor device: {c.device}")
        
        # Compare with CPU
        print("\n2. CPU vs GPU speed comparison...")
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        print(f"   CPU time: {cpu_time:.4f} seconds")
        print(f"   GPU time: {gpu_time:.4f} seconds")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Memory usage test
        print("\n3. GPU memory usage test...")
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate large tensor
        large_tensor = torch.randn(5000, 5000, device=device)
        after_allocation = torch.cuda.memory_allocated()
        
        print(f"   Initial GPU memory: {initial_memory / (1024**2):.2f} MB")
        print(f"   After allocation: {after_allocation / (1024**2):.2f} MB")
        print(f"   Memory used: {(after_allocation - initial_memory) / (1024**2):.2f} MB")
        
        # Clean up
        del large_tensor, a, b, c
        torch.cuda.empty_cache()
        
        print("   ✅ GPU computation test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ GPU computation test failed: {e}")
        return False

def test_neural_network():
    """Test a simple neural network on GPU"""
    print("\n" + "="*60)
    print("Neural Network GPU Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Cannot test neural network - CUDA not available.")
        return False
    
    try:
        device = torch.device('cuda:0')
        
        # Simple neural network
        class SimpleNN(torch.nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc1 = torch.nn.Linear(784, 256)
                self.fc2 = torch.nn.Linear(256, 128)
                self.fc3 = torch.nn.Linear(128, 10)
                self.relu = torch.nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # Create model and move to GPU
        model = SimpleNN().to(device)
        print(f"Model created and moved to: {device}")
        
        # Create dummy data
        batch_size = 64
        input_data = torch.randn(batch_size, 784, device=device)
        target = torch.randint(0, 10, (batch_size,), device=device)
        
        # Forward pass
        output = model(input_data)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        # Loss and backward pass
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Training step completed. Loss: {loss.item():.4f}")
        print("✅ Neural network GPU test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False

def main():
    """Main function to run all tests"""
    print("RTX 5070 PyTorch GPU Compatibility Test")
    print("This script will test if PyTorch can utilize your RTX 5070 GPU")
    print()
    
    # Run all tests
    tests_passed = 0
    total_tests = 4
    
    if check_pytorch_installation():
        tests_passed += 1
    
    if check_gpu_details():
        tests_passed += 1
    
    if test_gpu_computation():
        tests_passed += 1
        
    if test_neural_network():
        tests_passed += 1
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 SUCCESS! PyTorch is working correctly with your GPU.")
        print("Your RTX 5070 is ready for deep learning tasks!")
    elif tests_passed >= 2:
        print("⚠️  PARTIAL SUCCESS: Basic GPU functionality works.")
        print("Some advanced features may need troubleshooting.")
    else:
        print("❌ FAILURE: PyTorch cannot properly utilize your GPU.")
        print("Please check your CUDA installation and drivers.")
    
    print("\nNext steps:")
    print("- If successful: You can start training neural networks!")
    print("- If failed: Check NVIDIA drivers, CUDA installation, and PyTorch version")

if __name__ == "__main__":
    main()