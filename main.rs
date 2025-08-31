use crossbeam_channel::{Receiver, Sender, unbounded};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use tracing::{Level, Span, span};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use vulkano::VulkanLibrary;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
};
use vulkano::sync::GpuFuture;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const ITERATIONS: u32 = 256;
const WORKER_THREADS: usize = 8;
const COMPUTE_THREADS: usize = 4;
const TILE_SIZE: u32 = 64; // Each tile is 64x64 pixels
const TILES_X: u32 = (WIDTH + TILE_SIZE - 1) / TILE_SIZE;
const TILES_Y: u32 = (HEIGHT + TILE_SIZE - 1) / TILE_SIZE;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct FractalParams {
    width: u32,
    height: u32,
    max_iterations: u32,
    zoom: f32,
    center_x: f32,
    center_y: f32,
    julia_c_real: f32,
    julia_c_imag: f32,
    time: f32,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            width: WIDTH,
            height: HEIGHT,
            max_iterations: ITERATIONS,
            zoom: 1.0,
            center_x: 0.0,
            center_y: 0.0,
            julia_c_real: -0.4,
            julia_c_imag: 0.6,
            time: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
enum WorkerMessage {
    ComputeFrame(u32, FractalParams, Option<Span>),
    ComputeTile(TileWork),
    ComputeTileWithDeps(TileWork, Vec<u32>), // tile_work, dependency_tile_ids
    ComputePass(PassWork),                   // Multi-pass rendering
    PreprocessData(Vec<f32>, Option<Span>),
    PostprocessData(Vec<u8>, Option<Span>),
    Synchronize(u32),
    Shutdown,
}

#[derive(Debug, Clone)]
struct PassWork {
    pass_id: u32,
    frame_id: u32,
    pass_type: PassType,
    input_data: Vec<[u8; 4]>,
    params: FractalParams,
    parent_span: Option<Span>,
}

#[derive(Debug, Clone)]
enum PassType {
    BaseRender,
    AntiAliasing,
    PostProcessing,
    FinalComposite,
}

#[derive(Debug, Clone)]
struct TileWork {
    tile_id: u32,
    frame_id: u32,
    x_start: u32,
    y_start: u32,
    width: u32,
    height: u32,
    params: FractalParams,
    parent_span: Option<Span>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum WorkerResult {
    FrameReady(u32, FractalParams),
    TileComplete(TileResult),
    PassComplete(PassResult),
    DataProcessed(Vec<f32>),
    ImageProcessed(Vec<u8>),
    SyncComplete(u32),
}

#[derive(Debug, Clone)]
struct PassResult {
    pass_id: u32,
    frame_id: u32,
    pass_type: PassType,
    output_data: Vec<[u8; 4]>,
}

#[derive(Debug, Clone)]
struct TileResult {
    tile_id: u32,
    frame_id: u32,
    x_start: u32,
    y_start: u32,
    width: u32,
    height: u32,
    pixel_data: Vec<[u8; 4]>, // RGBA pixels
    edge_values: Vec<f32>,    // Edge values for neighboring tile dependencies
}

// CPU-based fractal computation functions
fn compute_julia_fractal_tile(
    tile_work: &TileWork,
    dependency_edges: Option<&HashMap<u32, Vec<f32>>>,
) -> TileResult {
    let mut pixel_data = Vec::with_capacity((tile_work.width * tile_work.height) as usize);
    let mut edge_values = Vec::new();

    let aspect_ratio = WIDTH as f32 / HEIGHT as f32;
    let c_real = tile_work.params.julia_c_real + 0.1 * (tile_work.params.time).sin();
    let c_imag = tile_work.params.julia_c_imag + 0.1 * (tile_work.params.time * 0.7).cos();

    for y in 0..tile_work.height {
        for x in 0..tile_work.width {
            let pixel_x = tile_work.x_start + x;
            let pixel_y = tile_work.y_start + y;

            // Convert to complex plane coordinates
            let real = (pixel_x as f32 / WIDTH as f32 - 0.5) * 4.0 / tile_work.params.zoom
                * aspect_ratio
                + tile_work.params.center_x;
            let imag = (pixel_y as f32 / HEIGHT as f32 - 0.5) * 4.0 / tile_work.params.zoom
                + tile_work.params.center_y;

            // Apply dependency influence if available
            let (z_real, z_imag) = if let Some(deps) = dependency_edges {
                apply_dependency_influence(real, imag, deps, tile_work.tile_id)
            } else {
                (real, imag)
            };

            let (iteration, smooth_iter, trap_min) = julia_iteration(
                z_real,
                z_imag,
                c_real,
                c_imag,
                tile_work.params.max_iterations,
            );

            let color = compute_fractal_color(
                iteration,
                smooth_iter,
                trap_min,
                tile_work.params.max_iterations,
                tile_work.params.time,
            );
            pixel_data.push(color);

            // Store edge values for dependencies
            if is_edge_pixel(x, y, tile_work.width, tile_work.height) {
                edge_values.push(smooth_iter);
            }
        }
    }

    TileResult {
        tile_id: tile_work.tile_id,
        frame_id: tile_work.frame_id,
        x_start: tile_work.x_start,
        y_start: tile_work.y_start,
        width: tile_work.width,
        height: tile_work.height,
        pixel_data,
        edge_values,
    }
}

fn julia_iteration(
    mut z_real: f32,
    mut z_imag: f32,
    c_real: f32,
    c_imag: f32,
    max_iterations: u32,
) -> (u32, f32, f32) {
    let mut iteration = 0;
    let mut trap_min: f32 = 1000.0;

    for i in 0..max_iterations {
        let z_real_squared = z_real * z_real;
        let z_imag_squared = z_imag * z_imag;

        // Orbit trap for additional coloring
        let dist = ((z_real - 0.5) * (z_real - 0.5) + z_imag * z_imag).sqrt();
        trap_min = trap_min.min(dist);

        if z_real_squared + z_imag_squared > 4.0 {
            let smooth_iter =
                i as f32 + 1.0 - (z_real_squared + z_imag_squared).sqrt().ln().ln() / 2.0_f32.ln();
            return (i, smooth_iter, trap_min);
        }

        let new_real = z_real_squared - z_imag_squared + c_real;
        let new_imag = 2.0 * z_real * z_imag + c_imag;

        z_real = new_real;
        z_imag = new_imag;
        iteration = i;
    }

    (iteration, iteration as f32, trap_min)
}

fn apply_dependency_influence(
    real: f32,
    imag: f32,
    deps: &HashMap<u32, Vec<f32>>,
    _tile_id: u32,
) -> (f32, f32) {
    let mut influence_real = 0.0;
    let mut influence_imag = 0.0;
    let mut count = 0;

    // Average influence from neighboring tiles
    for edge_values in deps.values() {
        if !edge_values.is_empty() {
            let avg = edge_values.iter().sum::<f32>() / edge_values.len() as f32;
            influence_real += avg * 0.01; // Small influence factor
            influence_imag += avg * 0.005;
            count += 1;
        }
    }

    if count > 0 {
        influence_real /= count as f32;
        influence_imag /= count as f32;
    }

    (real + influence_real, imag + influence_imag)
}

fn compute_fractal_color(
    iteration: u32,
    smooth_iter: f32,
    trap_min: f32,
    max_iterations: u32,
    time: f32,
) -> [u8; 4] {
    if iteration == max_iterations {
        let interior = trap_min * 3.0;
        let r = (interior * 0.1 * 255.0) as u8;
        let g = (interior * 0.2 * 255.0) as u8;
        let b = (interior * 0.3 * 255.0) as u8;
        [r, g, b, 255]
    } else {
        let hue = smooth_iter / max_iterations as f32 * 3.0 + time * 0.1 + trap_min;
        let saturation = 0.8 - trap_min * 0.3;
        let value = 1.0 - (smooth_iter / max_iterations as f32).powf(0.5);

        let rgb = hsv_to_rgb(hue, saturation, value);
        [rgb.0, rgb.1, rgb.2, 255]
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h - h.floor(); // Normalize hue to [0, 1)
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn is_edge_pixel(x: u32, y: u32, width: u32, height: u32) -> bool {
    x == 0 || y == 0 || x == width - 1 || y == height - 1
}

// Multi-pass rendering functions
fn compute_rendering_pass(pass_work: &PassWork) -> PassResult {
    match pass_work.pass_type {
        PassType::BaseRender => {
            // This would typically be handled by tile computation
            PassResult {
                pass_id: pass_work.pass_id,
                frame_id: pass_work.frame_id,
                pass_type: pass_work.pass_type.clone(),
                output_data: pass_work.input_data.clone(),
            }
        }
        PassType::AntiAliasing => {
            // Apply anti-aliasing filter
            let aa_data = apply_antialiasing(&pass_work.input_data);
            PassResult {
                pass_id: pass_work.pass_id,
                frame_id: pass_work.frame_id,
                pass_type: pass_work.pass_type.clone(),
                output_data: aa_data,
            }
        }
        PassType::PostProcessing => {
            // Apply post-processing effects like bloom, contrast, etc.
            let pp_data = apply_post_processing(&pass_work.input_data, pass_work.params.time);
            PassResult {
                pass_id: pass_work.pass_id,
                frame_id: pass_work.frame_id,
                pass_type: pass_work.pass_type.clone(),
                output_data: pp_data,
            }
        }
        PassType::FinalComposite => {
            // Final composite pass
            let composite_data = apply_final_composite(&pass_work.input_data);
            PassResult {
                pass_id: pass_work.pass_id,
                frame_id: pass_work.frame_id,
                pass_type: pass_work.pass_type.clone(),
                output_data: composite_data,
            }
        }
    }
}

fn apply_antialiasing(input: &[[u8; 4]]) -> Vec<[u8; 4]> {
    // Simple box blur anti-aliasing
    let mut output = input.to_vec();
    let width = WIDTH as usize;
    let height = HEIGHT as usize;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;

            // 3x3 kernel
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let nidx = ny * width + nx;

                    if nidx < input.len() {
                        r_sum += input[nidx][0] as u32;
                        g_sum += input[nidx][1] as u32;
                        b_sum += input[nidx][2] as u32;
                    }
                }
            }

            output[idx] = [(r_sum / 9) as u8, (g_sum / 9) as u8, (b_sum / 9) as u8, 255];
        }
    }

    output
}

fn apply_post_processing(input: &[[u8; 4]], time: f32) -> Vec<[u8; 4]> {
    // Apply time-based color effects
    input
        .iter()
        .map(|pixel| {
            let brightness_mod = 1.0 + (time * 0.5).sin() * 0.1;
            let contrast_mod = 1.1;

            let r = (pixel[0] as f32 * brightness_mod * contrast_mod).min(255.0) as u8;
            let g = (pixel[1] as f32 * brightness_mod * contrast_mod).min(255.0) as u8;
            let b = (pixel[2] as f32 * brightness_mod * contrast_mod).min(255.0) as u8;

            [r, g, b, pixel[3]]
        })
        .collect()
}

fn apply_final_composite(input: &[[u8; 4]]) -> Vec<[u8; 4]> {
    // Final gamma correction
    input
        .iter()
        .map(|pixel| {
            let r = ((pixel[0] as f32 / 255.0).powf(1.0 / 2.2) * 255.0) as u8;
            let g = ((pixel[1] as f32 / 255.0).powf(1.0 / 2.2) * 255.0) as u8;
            let b = ((pixel[2] as f32 / 255.0).powf(1.0 / 2.2) * 255.0) as u8;

            [r, g, b, pixel[3]]
        })
        .collect()
}

struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: Sender<WorkerMessage>,
    result_receiver: Receiver<WorkerResult>,
    barrier: Arc<Barrier>,
    shutdown: Arc<AtomicBool>,
    #[allow(dead_code)]
    work_counter: Arc<AtomicU64>,
    completed_tiles: Arc<RwLock<HashMap<u32, HashMap<u32, TileResult>>>>, // frame_id -> tile_id -> result
}

impl ThreadPool {
    fn new(num_threads: usize) -> Self {
        let (work_sender, work_receiver) = unbounded::<WorkerMessage>();
        let (result_sender, result_receiver) = unbounded::<WorkerResult>();
        let work_receiver = Arc::new(work_receiver);
        let barrier = Arc::new(Barrier::new(num_threads + 1));
        let shutdown = Arc::new(AtomicBool::new(false));
        let work_counter = Arc::new(AtomicU64::new(0));
        let completed_tiles = Arc::new(RwLock::new(HashMap::new()));

        let mut workers = Vec::new();

        for id in 0..num_threads {
            let work_receiver = Arc::clone(&work_receiver);
            let result_sender = result_sender.clone();
            let barrier = Arc::clone(&barrier);
            let shutdown = Arc::clone(&shutdown);
            let work_counter = Arc::clone(&work_counter);

            let handle = thread::spawn(move || {
                Self::worker_thread(
                    id,
                    work_receiver,
                    result_sender,
                    barrier,
                    shutdown,
                    work_counter,
                );
            });

            workers.push(handle);
        }

        ThreadPool {
            workers,
            sender: work_sender,
            result_receiver,
            barrier,
            shutdown,
            work_counter,
            completed_tiles,
        }
    }

    fn worker_thread(
        id: usize,
        receiver: Arc<Receiver<WorkerMessage>>,
        sender: Sender<WorkerResult>,
        barrier: Arc<Barrier>,
        shutdown: Arc<AtomicBool>,
        work_counter: Arc<AtomicU64>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            let msg = receiver.recv();

            match msg {
                Ok(WorkerMessage::ComputeFrame(frame_id, params, parent_span)) => {
                    let work_span = span!(
                        Level::INFO,
                        "compute_frame",
                        worker_id = id,
                        frame_id = frame_id
                    );

                    // Establish follows_from relationship if parent span exists
                    if let Some(parent) = parent_span {
                        work_span.follows_from(parent);
                    }

                    let _work_guard = work_span.enter();

                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // Simulate CPU-intensive preprocessing
                    let mut _sum = 0.0f32;
                    for i in 0..100000 {
                        _sum += (i as f32 * params.time).sin();
                    }

                    // Add some jitter to stress scheduler
                    thread::sleep(Duration::from_micros((id as u64 * 100) % 500));

                    sender
                        .send(WorkerResult::FrameReady(frame_id, params))
                        .unwrap();
                }
                Ok(WorkerMessage::ComputeTile(tile_work)) => {
                    let work_span = span!(
                        Level::INFO,
                        "compute_tile",
                        worker_id = id,
                        tile_id = tile_work.tile_id,
                        frame_id = tile_work.frame_id
                    );

                    if let Some(parent) = &tile_work.parent_span {
                        work_span.follows_from(parent.clone());
                    }

                    let _work_guard = work_span.enter();
                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // Compute fractal tile on CPU
                    let result = compute_julia_fractal_tile(&tile_work, None);
                    sender.send(WorkerResult::TileComplete(result)).unwrap();
                }
                Ok(WorkerMessage::ComputeTileWithDeps(tile_work, _dep_ids)) => {
                    let work_span = span!(
                        Level::INFO,
                        "compute_tile_with_deps",
                        worker_id = id,
                        tile_id = tile_work.tile_id,
                        frame_id = tile_work.frame_id
                    );

                    if let Some(parent) = &tile_work.parent_span {
                        work_span.follows_from(parent.clone());
                    }

                    let _work_guard = work_span.enter();
                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // TODO: Implement dependency resolution
                    // For now, compute without dependencies
                    let result = compute_julia_fractal_tile(&tile_work, None);
                    sender.send(WorkerResult::TileComplete(result)).unwrap();
                }
                Ok(WorkerMessage::PreprocessData(data, parent_span)) => {
                    let work_span = span!(
                        Level::INFO,
                        "preprocess_data",
                        worker_id = id,
                        data_size = data.len()
                    );

                    // Establish follows_from relationship if parent span exists
                    if let Some(parent) = parent_span {
                        work_span.follows_from(parent);
                    }

                    let _work_guard = work_span.enter();

                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // CPU-intensive data transformation
                    let processed: Vec<f32> = data
                        .iter()
                        .map(|&x| {
                            let mut result = x;
                            for _ in 0..100 {
                                result = (result * 1.1).sin() * 2.0;
                            }
                            result
                        })
                        .collect();

                    sender.send(WorkerResult::DataProcessed(processed)).unwrap();
                }
                Ok(WorkerMessage::PostprocessData(data, parent_span)) => {
                    let work_span = span!(
                        Level::INFO,
                        "postprocess_data",
                        worker_id = id,
                        data_size = data.len()
                    );

                    // Establish follows_from relationship if parent span exists
                    if let Some(parent) = parent_span {
                        work_span.follows_from(parent);
                    }

                    let _work_guard = work_span.enter();

                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // Simulate image post-processing
                    let processed: Vec<u8> = data
                        .iter()
                        .map(|&pixel| {
                            let mut p = pixel as f32 / 255.0;
                            p = p.powf(2.2); // Gamma correction
                            (p * 255.0) as u8
                        })
                        .collect();

                    sender
                        .send(WorkerResult::ImageProcessed(processed))
                        .unwrap();
                }
                Ok(WorkerMessage::ComputePass(pass_work)) => {
                    let work_span = span!(
                        Level::INFO,
                        "compute_pass",
                        worker_id = id,
                        pass_id = pass_work.pass_id,
                        frame_id = pass_work.frame_id
                    );

                    if let Some(parent) = &pass_work.parent_span {
                        work_span.follows_from(parent.clone());
                    }

                    let _work_guard = work_span.enter();
                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // Compute the rendering pass
                    let result = compute_rendering_pass(&pass_work);
                    sender.send(WorkerResult::PassComplete(result)).unwrap();
                }
                Ok(WorkerMessage::Synchronize(sync_id)) => {
                    let sync_span = span!(
                        Level::INFO,
                        "synchronize",
                        worker_id = id,
                        sync_id = sync_id
                    );
                    let _sync_guard = sync_span.enter();

                    barrier.wait();
                    sender.send(WorkerResult::SyncComplete(sync_id)).unwrap();
                }
                Ok(WorkerMessage::Shutdown) => {
                    break;
                }
                Err(_) => {
                    break;
                }
            }
        }
    }

    fn submit_work(&self, msg: WorkerMessage) {
        self.sender.send(msg).unwrap();
    }

    fn synchronize(&self, sync_id: u32) {
        for _ in 0..WORKER_THREADS {
            self.submit_work(WorkerMessage::Synchronize(sync_id));
        }
        self.barrier.wait();
    }

    #[allow(dead_code)]
    fn get_work_count(&self) -> u64 {
        self.work_counter.load(Ordering::Relaxed)
    }

    fn wait_for_frame_tiles(&self, frame_id: u32, expected_tiles: u32) -> HashMap<u32, TileResult> {
        let mut completed_count = 0;

        while completed_count < expected_tiles {
            if let Ok(result) = self
                .result_receiver
                .recv_timeout(Duration::from_millis(100))
            {
                if let WorkerResult::TileComplete(tile_result) = result {
                    if tile_result.frame_id == frame_id {
                        let mut tiles = self.completed_tiles.write().unwrap();
                        tiles
                            .entry(frame_id)
                            .or_insert_with(HashMap::new)
                            .insert(tile_result.tile_id, tile_result);
                        completed_count += 1;
                    }
                }
            }
        }

        // Return completed tiles for this frame
        let mut tiles = self.completed_tiles.write().unwrap();
        tiles.remove(&frame_id).unwrap_or_default()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for _ in 0..self.workers.len() {
            self.sender.send(WorkerMessage::Shutdown).ok();
        }
        for worker in self.workers.drain(..) {
            worker.join().ok();
        }
    }
}

struct ComputeScheduler {
    #[allow(dead_code)]
    thread_pool: Arc<ThreadPool>,
    compute_threads: Vec<thread::JoinHandle<()>>,
    frame_queue: Arc<RwLock<VecDeque<(u32, FractalParams, Option<Span>)>>>,
    shutdown: Arc<AtomicBool>,
}

impl ComputeScheduler {
    fn new(thread_pool: Arc<ThreadPool>) -> Self {
        let frame_queue = Arc::new(RwLock::new(VecDeque::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut compute_threads = Vec::new();

        for id in 0..COMPUTE_THREADS {
            let pool = Arc::clone(&thread_pool);
            let queue = Arc::clone(&frame_queue);
            let shutdown = Arc::clone(&shutdown);

            let handle = thread::spawn(move || {
                Self::compute_thread(id, pool, queue, shutdown);
            });

            compute_threads.push(handle);
        }

        ComputeScheduler {
            thread_pool,
            compute_threads,
            frame_queue,
            shutdown,
        }
    }

    fn compute_thread(
        id: usize,
        thread_pool: Arc<ThreadPool>,
        frame_queue: Arc<RwLock<VecDeque<(u32, FractalParams, Option<Span>)>>>,
        shutdown: Arc<AtomicBool>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            let work = {
                let mut queue = frame_queue.write().unwrap();
                queue.pop_front()
            };

            if let Some((frame_id, params, parent_span)) = work {
                let schedule_span = span!(
                    Level::INFO,
                    "schedule_frame",
                    compute_id = id,
                    frame_id = frame_id
                );
                let _schedule_guard = schedule_span.enter();

                // Submit preprocessing work to thread pool with parent span
                let preprocess_data = vec![params.time; 1000];
                thread_pool.submit_work(WorkerMessage::PreprocessData(
                    preprocess_data,
                    parent_span.clone(),
                ));

                // Use tile-based computation - schedule all tiles for this frame
                Self::schedule_tiles_for_frame(
                    Arc::clone(&thread_pool),
                    frame_id,
                    params,
                    parent_span.clone(),
                );

                // Simulate complex scheduling patterns
                if frame_id % 5 == 0 {
                    thread_pool.synchronize(frame_id);
                }

                thread::sleep(Duration::from_millis(1));
            } else {
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    fn schedule_frame(&self, frame_id: u32, params: FractalParams, parent_span: Option<Span>) {
        let mut queue = self.frame_queue.write().unwrap();
        queue.push_back((frame_id, params, parent_span));
    }

    fn schedule_tiles_for_frame(
        thread_pool: Arc<ThreadPool>,
        frame_id: u32,
        params: FractalParams,
        parent_span: Option<Span>,
    ) {
        // Create tiles for the frame
        let mut tile_id = 0;

        for tile_y in 0..TILES_Y {
            for tile_x in 0..TILES_X {
                let x_start = tile_x * TILE_SIZE;
                let y_start = tile_y * TILE_SIZE;
                let width = (TILE_SIZE).min(WIDTH - x_start);
                let height = (TILE_SIZE).min(HEIGHT - y_start);

                let tile_work = TileWork {
                    tile_id,
                    frame_id,
                    x_start,
                    y_start,
                    width,
                    height,
                    params,
                    parent_span: parent_span.clone(),
                };

                // For now, submit all tiles without dependencies
                // TODO: Add dependency logic for neighboring tiles
                thread_pool.submit_work(WorkerMessage::ComputeTile(tile_work));

                tile_id += 1;
            }
        }
    }
}

impl Drop for ComputeScheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for thread in self.compute_threads.drain(..) {
            thread.join().ok();
        }
    }
}

struct VulkanRenderer {
    #[allow(dead_code)]
    instance: Arc<Instance>,
    #[allow(dead_code)]
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    compute_pipeline: Arc<ComputePipeline>,
    output_image: Arc<ImageView>,
    cpu_data_image: Arc<ImageView>,
    params_buffer: Subbuffer<FractalParams>,
    thread_pool: Arc<ThreadPool>,
    scheduler: ComputeScheduler,
    #[allow(dead_code)]
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<ImageView>>,
}

impl VulkanRenderer {
    fn new(event_loop: &EventLoop<()>) -> Result<(Self, Arc<Window>), Box<dyn std::error::Error>> {
        let span = span!(Level::INFO, "vulkan_init");
        let _enter = span.enter();

        // Create thread pool first
        let thread_pool = Arc::new(ThreadPool::new(WORKER_THREADS));

        // Create compute scheduler
        let scheduler = ComputeScheduler::new(Arc::clone(&thread_pool));

        // Create window
        let window = Arc::new(
            WindowBuilder::new()
                .with_title("Multi-threaded GPU Fractal Renderer")
                .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
                .build(event_loop)?,
        );

        // Create Vulkan instance
        let library = VulkanLibrary::new()?;
        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Self::create_instance(&library, &required_extensions)?;

        // Create surface
        let surface = Surface::from_window(instance.clone(), window.clone())?;

        // Select physical device
        let physical_device = Self::select_physical_device(&instance, &surface)?;

        // Create logical device and queue
        let (device, queue) = Self::create_device(physical_device, &surface)?;

        // Create allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Create compute pipeline
        let compute_pipeline = Self::create_compute_pipeline(device.clone())?;

        // Create output image and CPU data image with RGBA8 format for compute shader
        let output_image = Self::create_output_image(&memory_allocator)?;
        let cpu_data_image = Self::create_cpu_data_image(&memory_allocator)?;

        // Create swapchain
        let (swapchain, swapchain_images) = Self::create_swapchain(&device, &surface)?;

        // Create params buffer
        let params_buffer = Self::create_params_buffer(&memory_allocator)?;

        let renderer = Self {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            compute_pipeline,
            output_image,
            cpu_data_image,
            params_buffer,
            thread_pool,
            scheduler,
            surface,
            swapchain,
            swapchain_images,
        };

        Ok((renderer, window.clone()))
    }

    fn create_instance(
        library: &Arc<VulkanLibrary>,
        required_extensions: &vulkano::instance::InstanceExtensions,
    ) -> Result<Arc<Instance>, Box<dyn std::error::Error>> {
        let create_info = InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: *required_extensions,
            application_name: Some("Multi-threaded GPU Fractal Renderer".into()),
            application_version: vulkano::Version::V1_0,
            engine_name: Some("Fractal Engine MT".into()),
            engine_version: vulkano::Version::V1_0,
            ..Default::default()
        };

        let instance = Instance::new(library.clone(), create_info)?;

        Ok(instance)
    }

    fn select_physical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface>,
    ) -> Result<Arc<PhysicalDevice>, Box<dyn std::error::Error>> {
        let physical_devices = instance.enumerate_physical_devices()?;

        let physical_device = physical_devices
            .filter(|p| {
                let extensions_supported = p.supported_extensions().contains(&DeviceExtensions {
                    khr_storage_buffer_storage_class: true,
                    khr_swapchain: true,
                    ..Default::default()
                });
                let surface_supported = p.surface_support(0, surface).unwrap_or(false);
                let supported = extensions_supported && surface_supported;
                supported
            })
            .min_by_key(|p| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .ok_or("No suitable physical device found")?;

        Ok(physical_device)
    }

    fn create_device(
        physical_device: Arc<PhysicalDevice>,
        surface: &Arc<Surface>,
    ) -> Result<(Arc<Device>, Arc<Queue>), Box<dyn std::error::Error>> {
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
                    && physical_device
                        .surface_support(i as u32, surface)
                        .unwrap_or(false)
            })
            .ok_or("No suitable queue family found")? as u32;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions {
                    khr_storage_buffer_storage_class: true,
                    khr_swapchain: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;

        let queue = queues.next().ok_or("No queue created")?;

        Ok((device, queue))
    }

    fn create_compute_pipeline(
        device: Arc<Device>,
    ) -> Result<Arc<ComputePipeline>, Box<dyn std::error::Error>> {
        let shader = Self::load_shader(device.clone())?;

        let stage = PipelineShaderStageCreateInfo::new(shader.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())?,
        )?;

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )?;

        Ok(pipeline)
    }

    fn load_shader(device: Arc<Device>) -> Result<Arc<ShaderModule>, Box<dyn std::error::Error>> {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r#"
                    #version 450

                    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

                    layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
                    layout(set = 0, binding = 1, rgba8) uniform readonly image2D cpu_data;
                    
                    layout(set = 0, binding = 2) uniform FractalParams {
                        uint width;
                        uint height;
                        uint max_iterations;
                        float zoom;
                        float center_x;
                        float center_y;
                        float julia_c_real;
                        float julia_c_imag;
                        float time;
                    } params;

                    void main() {
                        ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

                        if (pixel_coords.x >= params.width || pixel_coords.y >= params.height) {
                            return;
                        }

                        // Simply copy the CPU-computed fractal data to output
                        vec4 cpu_color = imageLoad(cpu_data, pixel_coords);
                        imageStore(img, pixel_coords, cpu_color);
                    }
                "#
            }
        }

        Ok(cs::load(device)?)
    }

    fn create_output_image(
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Result<Arc<ImageView>, Box<dyn std::error::Error>> {
        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                extent: [WIDTH, HEIGHT, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let image_view = ImageView::new_default(image)?;

        Ok(image_view)
    }

    fn create_cpu_data_image(
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Result<Arc<ImageView>, Box<dyn std::error::Error>> {
        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                extent: [WIDTH, HEIGHT, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let image_view = ImageView::new_default(image)?;

        Ok(image_view)
    }

    fn create_params_buffer(
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Result<Subbuffer<FractalParams>, Box<dyn std::error::Error>> {
        let buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            FractalParams::default(),
        )?;

        Ok(buffer)
    }

    fn create_swapchain(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<ImageView>>), Box<dyn std::error::Error>> {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(surface, Default::default())?;
        let image_format = device
            .physical_device()
            .surface_formats(surface, Default::default())?[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: [WIDTH, HEIGHT],
                image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT
                    | vulkano::image::ImageUsage::TRANSFER_DST,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )?;

        let swapchain_images = images
            .into_iter()
            .map(|image| ImageView::new_default(image))
            .collect::<Result<Vec<_>, _>>()?;

        Ok((swapchain, swapchain_images))
    }

    pub fn render_frame(
        &mut self,
        frame_id: u32,
        params: FractalParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let span = span!(Level::INFO, "render_frame", frame_id = frame_id);
        let _enter = span.enter();

        // Schedule tile-based fractal computation
        ComputeScheduler::schedule_tiles_for_frame(
            Arc::clone(&self.thread_pool),
            frame_id,
            params,
            Some(span.clone()),
        );

        // MAIN THREAD DEPENDENCY: Wait for all tiles to complete before proceeding
        let total_tiles = TILES_X * TILES_Y;
        let completed_tiles = self.thread_pool.wait_for_frame_tiles(frame_id, total_tiles);

        // Composite tiles into final image buffer
        let base_image = self.composite_tiles(completed_tiles);

        // Multi-pass rendering with sequential dependencies
        let final_image = self.apply_multi_pass_rendering(frame_id, base_image, params)?;

        // Upload CPU-computed fractal data to GPU
        self.upload_cpu_data_to_gpu(&final_image)?;

        // Simulate some post-processing work
        let dummy_image_data = vec![128u8; 1000];
        self.thread_pool
            .submit_work(WorkerMessage::PostprocessData(dummy_image_data, None));

        // Update parameters
        {
            let mut content = self.params_buffer.write()?;
            *content = params;
        }

        // Acquire next swapchain image
        let (image_index, _suboptimal, acquire_future) =
            acquire_next_image(self.swapchain.clone(), None)?;
        let swapchain_image = &self.swapchain_images[image_index as usize];

        // Create descriptor set
        let layout = self.compute_pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.output_image.clone()),
                WriteDescriptorSet::image_view(1, self.cpu_data_image.clone()),
                WriteDescriptorSet::buffer(2, self.params_buffer.clone()),
            ],
            [],
        )?;

        // Build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )?
            .dispatch([(WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1])?
            .blit_image(vulkano::command_buffer::BlitImageInfo::images(
                self.output_image.image().clone(),
                swapchain_image.image().clone(),
            ))?;

        let command_buffer = builder.build()?;

        // Submit to GPU and present
        let future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()?;

        // Wait for completion
        future.wait(None)?;

        Ok(())
    }

    fn composite_tiles(&self, tiles: HashMap<u32, TileResult>) -> Vec<[u8; 4]> {
        let mut final_image = vec![[0u8; 4]; (WIDTH * HEIGHT) as usize];

        for tile_result in tiles.values() {
            let tile_pixels = &tile_result.pixel_data;
            let mut pixel_idx = 0;

            for y in 0..tile_result.height {
                for x in 0..tile_result.width {
                    let screen_x = tile_result.x_start + x;
                    let screen_y = tile_result.y_start + y;

                    if screen_x < WIDTH && screen_y < HEIGHT {
                        let screen_idx = (screen_y * WIDTH + screen_x) as usize;
                        if pixel_idx < tile_pixels.len() && screen_idx < final_image.len() {
                            final_image[screen_idx] = tile_pixels[pixel_idx];
                        }
                    }

                    pixel_idx += 1;
                }
            }
        }

        final_image
    }

    fn upload_cpu_data_to_gpu(
        &mut self,
        cpu_data: &[[u8; 4]],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create a staging buffer to upload the CPU data
        let staging_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            cpu_data
                .iter()
                .flat_map(|pixel| pixel.iter())
                .cloned()
                .collect::<Vec<u8>>(),
        )?;

        // Create a command buffer to copy data to the GPU image
        let mut cmd_builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        cmd_builder.copy_buffer_to_image(
            vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                staging_buffer,
                self.cpu_data_image.image().clone(),
            ),
        )?;

        let command_buffer = cmd_builder.build()?;
        let future = command_buffer
            .execute(self.queue.clone())?
            .then_signal_fence_and_flush()?;

        // Wait for upload to complete
        future.wait(None)?;

        Ok(())
    }

    fn apply_multi_pass_rendering(
        &mut self,
        frame_id: u32,
        base_image: Vec<[u8; 4]>,
        params: FractalParams,
    ) -> Result<Vec<[u8; 4]>, Box<dyn std::error::Error>> {
        // Pass 1: Anti-aliasing (depends on base render)
        let aa_pass = PassWork {
            pass_id: 1,
            frame_id,
            pass_type: PassType::AntiAliasing,
            input_data: base_image,
            params,
            parent_span: None,
        };

        self.thread_pool
            .submit_work(WorkerMessage::ComputePass(aa_pass));
        let aa_result = self.wait_for_pass_completion(frame_id, 1)?;

        // Pass 2: Post-processing (depends on anti-aliasing)
        let pp_pass = PassWork {
            pass_id: 2,
            frame_id,
            pass_type: PassType::PostProcessing,
            input_data: aa_result,
            params,
            parent_span: None,
        };

        self.thread_pool
            .submit_work(WorkerMessage::ComputePass(pp_pass));
        let pp_result = self.wait_for_pass_completion(frame_id, 2)?;

        // Pass 3: Final composite (depends on post-processing)
        let final_pass = PassWork {
            pass_id: 3,
            frame_id,
            pass_type: PassType::FinalComposite,
            input_data: pp_result,
            params,
            parent_span: None,
        };

        self.thread_pool
            .submit_work(WorkerMessage::ComputePass(final_pass));
        let final_result = self.wait_for_pass_completion(frame_id, 3)?;

        Ok(final_result)
    }

    fn wait_for_pass_completion(
        &self,
        frame_id: u32,
        pass_id: u32,
    ) -> Result<Vec<[u8; 4]>, Box<dyn std::error::Error>> {
        loop {
            if let Ok(result) = self
                .thread_pool
                .result_receiver
                .recv_timeout(Duration::from_millis(100))
            {
                if let WorkerResult::PassComplete(pass_result) = result {
                    if pass_result.frame_id == frame_id && pass_result.pass_id == pass_id {
                        return Ok(pass_result.output_data);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn benchmark(&mut self, frames: u32) -> Result<(), Box<dyn std::error::Error>> {
        let mut params = FractalParams::default();
        let start_time = Instant::now();

        // Spawn monitoring thread
        let work_counter = Arc::clone(&self.thread_pool.work_counter);
        let monitor_handle = thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(1));
                let current_count = work_counter.load(Ordering::Relaxed);
                if current_count > (frames as u64 * 3) {
                    break;
                }
            }
        });

        for frame in 0..frames {
            params.time = frame as f32 * 0.016; // Simulate 60 FPS timing
            params.zoom = 1.0 + (params.time * 0.1).sin() * 0.5;
            params.center_x = (params.time * 0.05).cos() * 0.3;
            params.center_y = (params.time * 0.03).sin() * 0.3;

            if frame % 10 == 0 {
                // Synchronize all threads periodically
                self.thread_pool.synchronize(frame);
            }

            self.render_frame(frame, params)?;

            // Add some thread contention
            if frame % 5 == 0 {
                let work_data = vec![frame as f32; 5000];
                self.thread_pool
                    .submit_work(WorkerMessage::PreprocessData(work_data, None));
            }
        }

        let elapsed = start_time.elapsed();
        let _fps = frames as f64 / elapsed.as_secs_f64();
        let _total_work = self.thread_pool.get_work_count();

        monitor_handle.join().ok();

        Ok(())
    }
}

fn setup_tracing() {
    let perfetto_layer = tracing_perfetto::PerfettoLayer::new(
        std::fs::File::create("trace.perfetto-trace").unwrap(),
    );
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file("trace-chrome.json")
        .build();

    // Enable both perfetto and chrome layers
    tracing_subscriber::registry()
        .with(perfetto_layer)
        .with(chrome_layer)
        .init();

    // Keep the guard alive for the duration of the program
    std::mem::forget(_guard);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_tracing();

    let event_loop = EventLoop::new();
    let (mut renderer, window) = VulkanRenderer::new(&event_loop)?;

    let mut frame_count = 0u32;
    let start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| -> () {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(winit::event::VirtualKeyCode::Escape) = input.virtual_keycode {
                        if input.state == winit::event::ElementState::Pressed {
                            *control_flow = winit::event_loop::ControlFlow::Exit;
                        }
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                let elapsed = start_time.elapsed().as_secs_f32();

                let mut params = FractalParams::default();
                params.time = elapsed;
                params.zoom = 1.0 + (elapsed * 0.1).sin() * 0.5;
                params.center_x = (elapsed * 0.05).cos() * 0.3;
                params.center_y = (elapsed * 0.03).sin() * 0.3;

                if let Err(_e) = renderer.render_frame(frame_count, params) {
                    // Silently handle render errors
                }

                frame_count += 1;
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                // Redraw handled in MainEventsCleared
            }
            _ => {}
        }
    });
}
