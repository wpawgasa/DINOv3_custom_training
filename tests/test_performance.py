"""
Performance benchmarks and stress tests for DINOv3 framework.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.augmentations import create_transforms
from data.dataset import create_dataset
from models.model_factory import create_model


class TestModelPerformance:
    """Performance benchmarks for model inference."""

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_inference_throughput(self, sample_model, batch_size):
        """Test inference throughput across different batch sizes."""
        sample_model.eval()
        device = next(sample_model.parameters()).device

        # Create test batch
        test_input = torch.randn(batch_size, 3, 224, 224, device=device)

        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = sample_model(test_input)

        # Benchmark runs
        num_runs = 50
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                outputs = sample_model(test_input)

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_runs
        samples_per_second = (batch_size * num_runs) / total_time

        # Store results
        results = {
            "batch_size": batch_size,
            "avg_time_per_batch": avg_time_per_batch,
            "samples_per_second": samples_per_second,
            "total_samples": batch_size * num_runs,
            "total_time": total_time,
        }

        print(
            f"Batch size {batch_size}: {samples_per_second:.1f} samples/sec, "
            f"{avg_time_per_batch*1000:.1f} ms/batch"
        )

        # Performance assertions
        assert avg_time_per_batch < 5.0  # Less than 5 seconds per batch
        assert samples_per_second > 0.1  # At least 0.1 samples per second

        # Larger batches should be more efficient per sample
        if batch_size > 1:
            single_sample_time = avg_time_per_batch / batch_size
            assert single_sample_time < 1.0  # Less than 1 second per sample

    @pytest.mark.slow
    @pytest.mark.parametrize("input_size", [(224, 224), (256, 256), (384, 384)])
    def test_inference_across_input_sizes(self, sample_model, input_size):
        """Test inference performance across different input sizes."""
        sample_model.eval()
        device = next(sample_model.parameters()).device

        height, width = input_size
        test_input = torch.randn(4, 3, height, width, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = sample_model(test_input)

        # Benchmark
        num_runs = 20
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                outputs = sample_model(test_input)

        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs

        print(f"Input size {input_size}: {avg_time*1000:.1f} ms/batch")

        # Larger inputs should take more time but not excessively
        assert avg_time < 10.0  # Less than 10 seconds

        # Memory usage should be reasonable
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Peak GPU memory: {memory_mb:.1f} MB")
            assert memory_mb < 16384  # Less than 16GB

    @pytest.mark.gpu
    def test_gpu_memory_efficiency(self, sample_model):
        """Test GPU memory usage patterns and efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        device = torch.device("cuda")
        model = sample_model.to(device)
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16]
        memory_stats = {}

        for batch_size in batch_sizes:
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            test_input = torch.randn(batch_size, 3, 224, 224, device=device)

            # Measure memory during inference
            with torch.no_grad():
                outputs = model(test_input)

            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            peak_memory = torch.cuda.max_memory_allocated()

            memory_stats[batch_size] = {
                "allocated_mb": memory_allocated / 1024 / 1024,
                "reserved_mb": memory_reserved / 1024 / 1024,
                "peak_mb": peak_memory / 1024 / 1024,
            }

            print(f"Batch {batch_size}: Peak {peak_memory/1024/1024:.1f} MB")

        # Memory should scale reasonably
        assert memory_stats[4]["peak_mb"] > memory_stats[1]["peak_mb"]
        assert memory_stats[16]["peak_mb"] > memory_stats[4]["peak_mb"]

        # But not linearly (due to fixed model size)
        ratio_4_1 = memory_stats[4]["peak_mb"] / memory_stats[1]["peak_mb"]
        assert ratio_4_1 < 4.0  # Should be less than 4x due to model overhead

        # Memory usage should be reasonable
        assert memory_stats[16]["peak_mb"] < 8192  # Less than 8GB for batch 16

    @pytest.mark.slow
    def test_cpu_vs_gpu_performance(self, model_config):
        """Compare CPU vs GPU performance."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        if len(devices) == 1:
            pytest.skip("GPU not available for comparison")

        results = {}

        for device_name in devices:
            device = torch.device(device_name)

            with pytest.mock.patch(
                "transformers.AutoModel.from_pretrained"
            ) as mock_pretrained:
                mock_model = pytest.mock.Mock()
                mock_model.config.hidden_size = 384
                mock_pretrained.return_value = mock_model

                model = create_model(model_config)
                model = model.to(device)
                model.eval()

                test_input = torch.randn(4, 3, 224, 224, device=device)

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(test_input)

                # Benchmark
                start_time = time.time()

                with torch.no_grad():
                    for _ in range(20):
                        outputs = model(test_input)

                end_time = time.time()

                avg_time = (end_time - start_time) / 20
                results[device_name] = avg_time

                print(f"{device_name.upper()}: {avg_time*1000:.1f} ms/batch")

        # GPU should be faster than CPU (if both available)
        if "cuda" in results and "cpu" in results:
            speedup = results["cpu"] / results["cuda"]
            print(f"GPU speedup: {speedup:.1f}x")
            assert speedup > 1.0  # GPU should be faster

    @pytest.mark.slow
    def test_memory_leak_detection(self, sample_model):
        """Test for memory leaks during repeated inference."""
        sample_model.eval()
        device = next(sample_model.parameters()).device

        test_input = torch.randn(8, 3, 224, 224, device=device)

        # Get baseline memory
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
            baseline_memory = torch.cuda.memory_allocated()
        else:
            baseline_memory = psutil.Process().memory_info().rss

        # Run many inference iterations
        num_iterations = 100

        with torch.no_grad():
            for i in range(num_iterations):
                outputs = sample_model(test_input)

                # Check memory periodically
                if i % 20 == 0:
                    if torch.cuda.is_available() and device.type == "cuda":
                        current_memory = torch.cuda.memory_allocated()
                    else:
                        current_memory = psutil.Process().memory_info().rss

                    memory_growth = current_memory - baseline_memory

                    # Memory growth should be minimal
                    if device.type == "cuda":
                        max_growth_mb = 100  # 100 MB
                        memory_growth_mb = memory_growth / 1024 / 1024
                    else:
                        max_growth_mb = 500  # 500 MB for CPU
                        memory_growth_mb = memory_growth / 1024 / 1024

                    print(f"Iteration {i}: Memory growth {memory_growth_mb:.1f} MB")
                    assert memory_growth_mb < max_growth_mb


class TestDataLoadingPerformance:
    """Performance benchmarks for data loading pipeline."""

    @pytest.mark.slow
    def test_dataloader_throughput(self, sample_dataset_dir):
        """Test data loading throughput."""
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=transform,
            cache_images=False,
        )

        # Test different numbers of workers
        worker_configs = [0, 2, 4]
        results = {}

        for num_workers in worker_configs:
            dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            start_time = time.time()

            for batch_idx, (images, labels) in enumerate(dataloader):
                pass  # Just iterate through data

            end_time = time.time()

            total_time = end_time - start_time
            samples_per_second = len(dataset) / total_time

            results[num_workers] = {
                "total_time": total_time,
                "samples_per_second": samples_per_second,
            }

            print(f"Workers {num_workers}: {samples_per_second:.1f} samples/sec")

        # More workers should generally be faster (up to a point)
        assert results[0]["samples_per_second"] > 0

        # Should achieve reasonable throughput
        best_throughput = max(r["samples_per_second"] for r in results.values())
        assert best_throughput > 1.0  # At least 1 sample per second

    @pytest.mark.slow
    def test_augmentation_performance(self, sample_dataset_dir):
        """Test performance impact of different augmentation strategies."""
        augmentation_configs = [
            ("minimal", {"resize_shortest": True, "center_crop": True}),
            ("basic", {"horizontal_flip": True, "color_jitter": {"brightness": 0.2}}),
            (
                "advanced",
                {
                    "horizontal_flip": True,
                    "color_jitter": {
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "saturation": 0.2,
                        "hue": 0.1,
                    },
                    "random_rotation": 10,
                    "gaussian_blur": 0.1,
                },
            ),
        ]

        results = {}

        for aug_name, aug_config in augmentation_configs:
            transform_manager = create_transforms(
                domain="natural", image_size=224, train_kwargs=aug_config, val_kwargs={}
            )

            dataset = create_dataset(
                data_path=str(sample_dataset_dir),
                annotation_format="imagefolder",
                transform=transform_manager.get_train_transform(),
                cache_images=False,
            )

            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

            start_time = time.time()

            for images, labels in dataloader:
                pass

            end_time = time.time()

            total_time = end_time - start_time
            samples_per_second = len(dataset) / total_time

            results[aug_name] = samples_per_second
            print(f"{aug_name} augmentations: {samples_per_second:.1f} samples/sec")

        # All configurations should achieve reasonable performance
        for throughput in results.values():
            assert throughput > 0.5  # At least 0.5 samples per second


class TestTrainingPerformance:
    """Performance benchmarks for training loops."""

    @pytest.mark.slow
    def test_training_step_performance(self, experiment_config, sample_dataset_dir):
        """Test training step performance."""
        experiment_config.data.train_data_path = str(sample_dataset_dir)
        experiment_config.data.batch_size = 8

        # Create training setup
        transform_manager = create_transforms(
            domain=experiment_config.augmentation.domain,
            image_size=experiment_config.data.image_size,
        )

        train_dataset = create_dataset(
            data_path=experiment_config.data.train_data_path,
            annotation_format="imagefolder",
            transform=transform_manager.get_train_transform(),
            cache_images=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=experiment_config.data.batch_size,
            shuffle=True,
            num_workers=0,
        )

        with pytest.mock.patch(
            "transformers.AutoModel.from_pretrained"
        ) as mock_pretrained:
            mock_model = pytest.mock.Mock()
            mock_model.config.hidden_size = 384
            mock_pretrained.return_value = mock_model

            model = create_model(experiment_config.model)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()

            # Benchmark training steps
            step_times = []

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 10:  # Test first 10 batches
                    break

                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets, _ = batch

                start_time = time.time()

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs["logits"], targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                end_time = time.time()
                step_time = end_time - start_time
                step_times.append(step_time)

            avg_step_time = np.mean(step_times)
            std_step_time = np.std(step_times)

            print(
                f"Average training step: {avg_step_time*1000:.1f} ± {std_step_time*1000:.1f} ms"
            )

            # Training steps should be reasonably fast
            assert avg_step_time < 5.0  # Less than 5 seconds per step
            assert std_step_time < avg_step_time  # Reasonable consistency

    @pytest.mark.slow
    def test_gradient_computation_performance(self, sample_model):
        """Test gradient computation performance."""
        sample_model.train()
        device = next(sample_model.parameters()).device

        batch_size = 8
        test_input = torch.randn(batch_size, 3, 224, 224, device=device)
        test_targets = torch.randint(
            0, sample_model.num_classes, (batch_size,), device=device
        )

        criterion = nn.CrossEntropyLoss()

        # Time forward + backward pass
        num_runs = 20
        forward_times = []
        backward_times = []

        for _ in range(num_runs):
            # Forward pass timing
            start_time = time.time()
            outputs = sample_model(test_input)
            loss = criterion(outputs["logits"], targets)
            forward_time = time.time() - start_time

            # Backward pass timing
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time

            forward_times.append(forward_time)
            backward_times.append(backward_time)

            # Clear gradients
            sample_model.zero_grad()

        avg_forward = np.mean(forward_times)
        avg_backward = np.mean(backward_times)

        print(f"Forward pass: {avg_forward*1000:.1f} ms")
        print(f"Backward pass: {avg_backward*1000:.1f} ms")

        # Both should be reasonably fast
        assert avg_forward < 2.0  # Less than 2 seconds
        assert avg_backward < 3.0  # Less than 3 seconds (usually slower than forward)


class TestScalabilityBenchmarks:
    """Scalability and stress tests."""

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_batch_size_scaling(self, sample_model, batch_size):
        """Test performance scaling with batch size."""
        sample_model.eval()
        device = next(sample_model.parameters()).device

        try:
            test_input = torch.randn(batch_size, 3, 224, 224, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = sample_model(test_input)

            # Benchmark
            start_time = time.time()

            with torch.no_grad():
                for _ in range(10):
                    outputs = sample_model(test_input)

            end_time = time.time()

            total_time = end_time - start_time
            time_per_sample = total_time / (10 * batch_size)

            print(f"Batch {batch_size}: {time_per_sample*1000:.2f} ms/sample")

            # Time per sample should decrease with larger batches (efficiency)
            assert time_per_sample < 1.0  # Less than 1 second per sample

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory for batch size {batch_size}")
            else:
                raise

    @pytest.mark.slow
    def test_long_running_stability(self, sample_model):
        """Test stability during long running inference."""
        sample_model.eval()
        device = next(sample_model.parameters()).device

        test_input = torch.randn(4, 3, 224, 224, device=device)

        # Run for extended period
        num_iterations = 200
        times = []

        with torch.no_grad():
            for i in range(num_iterations):
                start_time = time.time()
                outputs = sample_model(test_input)
                end_time = time.time()

                times.append(end_time - start_time)

                # Check for performance degradation
                if i > 50 and i % 50 == 0:
                    recent_avg = np.mean(times[-50:])
                    initial_avg = np.mean(times[:50])

                    degradation = recent_avg / initial_avg
                    print(f"Iteration {i}: Performance ratio {degradation:.2f}")

                    # Should not degrade significantly
                    assert degradation < 1.5  # Less than 50% degradation

        # Overall performance should be stable
        first_half_avg = np.mean(times[: num_iterations // 2])
        second_half_avg = np.mean(times[num_iterations // 2 :])

        performance_ratio = second_half_avg / first_half_avg
        assert performance_ratio < 1.2  # Less than 20% degradation

    @pytest.mark.slow
    def test_concurrent_inference(self, sample_model):
        """Test performance under concurrent inference requests."""
        import queue
        import threading

        sample_model.eval()
        device = next(sample_model.parameters()).device

        # Shared queue for results
        results_queue = queue.Queue()

        def inference_worker(worker_id, num_requests):
            times = []

            for i in range(num_requests):
                test_input = torch.randn(2, 3, 224, 224, device=device)

                start_time = time.time()
                with torch.no_grad():
                    outputs = sample_model(test_input)
                end_time = time.time()

                times.append(end_time - start_time)

            results_queue.put(
                {"worker_id": worker_id, "times": times, "avg_time": np.mean(times)}
            )

        # Create multiple workers
        num_workers = 3
        requests_per_worker = 20

        threads = []
        start_time = time.time()

        for i in range(num_workers):
            thread = threading.Thread(
                target=inference_worker, args=(i, requests_per_worker)
            )
            threads.append(thread)
            thread.start()

        # Wait for all workers
        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        # Collect results
        worker_results = []
        while not results_queue.empty():
            worker_results.append(results_queue.get())

        total_requests = num_workers * requests_per_worker
        overall_throughput = total_requests / total_time

        print(f"Concurrent throughput: {overall_throughput:.1f} requests/sec")

        # Should handle concurrent requests reasonably
        assert len(worker_results) == num_workers
        assert overall_throughput > 1.0  # At least 1 request per second

        # Individual workers should not be excessively slow
        for result in worker_results:
            assert result["avg_time"] < 5.0  # Less than 5 seconds per request
