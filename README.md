# Multi-threaded GPU Fractal Renderer

**Demo project for Linux Tracing Summit 2025 talk**: [*"Unifying Performance Analysis: Using Perfetto UI to Merge Multiple Linux Tracing Sources"*](https://cfp.tracingsummit.org/ts2025/talk/TKVU8W/)

This fractal renderer demonstrates complex multi-threaded CPU/GPU workloads while generating comprehensive tracing data for performance analysis with the Perfetto UI.

## 🎯 Purpose

This project serves as a practical demo for demonstrating how the **Perfetto UI** can be used to analyze and visualize performance data from multiple, disparate tracing sources on a single timeline. The fractal renderer intentionally creates interesting performance patterns across:

- **Multi-threaded CPU computation** (tile-based parameter generation)
- **GPU compute shaders** (Vulkan-based fractal rendering)
- **Complex synchronization patterns** (worker thread pools, adaptive quality systems)
- **Memory management** (CPU/GPU data transfers)

## 🔧 Architecture Overview

The renderer implements a **hybrid CPU/GPU architecture**:

- **Main Thread**: Window management, event handling, frame orchestration
- **Worker Thread Pool**: Tile-based parameter computation with complex dependencies
- **GPU Compute Pipeline**: Vulkan compute shaders for fractal generation
- **Adaptive Quality System**: Dynamic performance optimization based on frame timing

This creates rich, multi-layered performance data perfect for demonstrating Perfetto's visualization capabilities.

## 📊 Generated Tracing Data

The project generates multiple types of performance traces:

### 1. **Application Tracing** (`fractal.pftrace`)
- Custom instrumentation using Rust's `tracing` crate
- Hierarchical span relationships showing work distribution
- Detailed timing of CPU compute phases, GPU synchronization, and quality adaptation

### 2. **CPU Profiling** (`fractal.perftext`) 
- Linux `perf` CPU sampling at 997 Hz
- Call stack traces with DWARF debugging information
- Hotspot analysis across all threads

### 3. **Scheduler Traces** (`fractal.sched`)
- Linux kernel scheduler events via ftrace
- Thread wake-up patterns and scheduling decisions
- Context switches and CPU migration events

### 4. **Flame Graph** (`fractal.svg`)
- Traditional flame graph for quick CPU hotspot identification
- Generated from perf data for supplementary analysis

## 🚀 Quick Start

### Prerequisites

- **Linux system** with Vulkan support
- **Rust toolchain** (1.70+)
- **perf tools** for CPU profiling
- **Root access** (optional, for scheduler traces)

### Build and Run

```bash
# Basic execution
./run-linux.sh

# With tracing enabled
./collect-traces.sh
```

The `collect-traces.sh` script will:
1. Build the renderer in release mode
2. Start comprehensive trace collection
3. Run the program for 30 seconds
4. Generate all trace files automatically
5. Provide next steps for analysis

### Manual Tracing

```bash
# Build
cargo build --release

# Run with custom trace file
./target/release/fractal-renderer-vk --trace my-trace.pftrace
```

## 📈 Performance Analysis Workflow

This demo follows the exact workflow presented in the Linux Tracing Summit talk:

### Step 1: CPU Profile Analysis
- Load `fractal.perftext.gz` in Perfetto UI
- Identify CPU hotspots using timeline view
- Analyze per-thread CPU utilization patterns

### Step 2: Scheduler Investigation  
- Load `fractal.sched.gz` to see kernel scheduler behavior
- Correlate thread wake-ups with application activity
- Identify potential scheduling bottlenecks

### Step 3: Application Context
- Load `fractal.pftrace.gz` for application-level instrumentation
- See structured spans showing work distribution
- Understand application logic flow and timing

### Step 4: **Unified Timeline** 🎯
- **Load multiple files simultaneously** in Perfetto UI
- See CPU, kernel, and application events on the same timeline
- Discover insights impossible to see from individual traces

## 🎨 What the Renderer Does

The fractal renderer generates animated Julia set fractals with:

- **1920×1080 resolution** at target 60 FPS
- **Tile-based processing** (64×64 pixel tiles)
- **8 worker threads** for CPU parameter computation
- **Adaptive quality system** that adjusts detail based on performance
- **GPU compute shaders** for final fractal generation
- **Real-time animation** with time-based parameter evolution

## 📁 Project Structure

```
fractal-renderer-vk/
├── main.rs              # Core application logic
├── Cargo.toml          # Rust dependencies
├── collect-traces.sh   # Comprehensive trace collection
├── run-linux.sh        # Simple execution script
├── run-mac.sh          # macOS execution script
└── traces/             # Generated trace files
    ├── fractal.pftrace.gz    # Perfetto application traces
    ├── fractal.perftext.gz   # perf CPU profiles  
    ├── fractal.sched.gz      # Linux scheduler traces
    └── fractal.svg           # Flame graph visualization
```

## 🎯 Talk Demonstration Points

This demo specifically showcases:

1. **Multi-source trace correlation** - seeing CPU, scheduler, and application data together
2. **Timeline-based analysis** - understanding event causality across subsystems
3. **Performance bottleneck identification** - using multiple data sources to pinpoint issues
4. **Real-world complexity** - non-trivial threading and GPU workloads

## 🔧 Technical Details

### Dependencies
- **vulkano**: Vulkan API bindings for GPU compute
- **winit**: Cross-platform windowing
- **tracing**: Structured application instrumentation
- **tracing-perfetto**: Perfetto trace format export
- **crossbeam-channel**: High-performance thread communication

### Tracing Integration
The application uses Rust's `tracing` ecosystem with a custom Perfetto exporter, generating traces compatible with the Perfetto UI's native format for optimal visualization.

### Performance Issues for Analysis
The codebase intentionally includes interesting performance patterns for educational purposes. For example, commit [`b9e6764`](https://github.com/LalitMaganti/vulkan-tile-renderer-rs/commit/b9e6764) demonstrates a classic lock contention fix - moving expensive computation outside mutex locks to reduce contention between worker threads.

## 🎪 Live Demo Notes

During the talk, this demo will show:

1. **Loading individual traces** and their limitations
2. **Trace merging** and insight discovery
3. **Interactive exploration** of the unified timeline

## 📚 Further Reading

- [Perfetto Documentation](https://docs.perfetto.dev/)
- [Linux Tracing Summit 2025](https://tracingsummit.org/)
