use crossbeam_channel::{Receiver, Sender, unbounded};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use tracing::{Id, Level, span};
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
            julia_c_real: -0.7,
            julia_c_imag: 0.0,
            time: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
enum WorkerMessage {
    ComputeTile(TileWork),
    PreprocessData(Vec<f32>, Option<Id>),
    #[allow(dead_code)] // Used for worker synchronization
    Synchronize(u32),
    Shutdown,
}

#[derive(Debug, Clone)]
struct TileWork {
    tile_id: u32,
    frame_id: u32,
    params: FractalParams,
    parent_span_id: Option<Id>,
}

#[derive(Debug)]
enum WorkerResult {
    TileComplete(TileResult),
    #[allow(dead_code)] // Sent but handled generically in result processing
    DataProcessed(Vec<f32>),
    #[allow(dead_code)] // Sent but handled generically in result processing
    SyncComplete(u32),
}

#[derive(Debug, Clone)]
struct TileResult {
    tile_id: u32,
    frame_id: u32,
    pixel_data: Vec<[u8; 4]>, // RGBA pixels
}

// CPU-based parameter computation (simulates expensive work)
fn compute_tile_parameters(
    tile_work: &TileWork,
    _dependency_edges: Option<&HashMap<u32, Vec<f32>>>,
) -> TileResult {
    // Generate influence parameters instead of full pixel data
    // This simulates expensive CPU computation but produces less data
    let mut pixel_data = Vec::with_capacity(1); // Just one value per tile

    // Simulate expensive CPU work
    let mut influence_r = 0.0f32;
    let mut influence_g = 0.0f32;
    let mut influence_b = 0.0f32;

    // Complex computation to generate parameters
    for i in 0..10000 {
        let x = (tile_work.tile_id as f32 + i as f32) * 0.001;
        influence_r += (x * tile_work.params.time * 0.4).sin();
        influence_g += (x * tile_work.params.time * 0.5).cos();
        influence_b += (x * tile_work.params.time * 0.3).sin() * (x * 2.1).cos();
    }

    influence_r = (influence_r / 10000.0).tanh();
    influence_g = (influence_g / 10000.0).tanh();
    influence_b = (influence_b / 10000.0).tanh();

    // Store just one color per tile as influence parameters
    let influence_color = [
        ((influence_r * 0.5 + 0.5) * 255.0) as u8,
        ((influence_g * 0.5 + 0.5) * 255.0) as u8,
        ((influence_b * 0.5 + 0.5) * 255.0) as u8,
        255,
    ];

    pixel_data.push(influence_color);

    TileResult {
        tile_id: tile_work.tile_id,
        frame_id: tile_work.frame_id,
        pixel_data,
    }
}

struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: Sender<WorkerMessage>,
    result_receiver: Receiver<WorkerResult>,
    #[allow(dead_code)] // Used by worker threads via Arc
    barrier: Arc<Barrier>,
    shutdown: Arc<AtomicBool>,
    #[allow(dead_code)] // Used by worker threads via Arc
    work_counter: Arc<AtomicU64>,
    completed_tiles: Arc<RwLock<HashMap<u32, HashMap<u32, TileResult>>>>, // frame_id -> tile_id -> result
    recalibration_trigger: Arc<(Mutex<bool>, Condvar)>, // System recalibration trigger
    #[allow(dead_code)] // Used by worker threads via Arc
    global_state: Arc<Mutex<f32>>, // Global application state
}

impl ThreadPool {
    fn new(num_threads: usize) -> Result<Self, std::io::Error> {
        let (work_sender, work_receiver) = unbounded::<WorkerMessage>();
        let (result_sender, result_receiver) = unbounded::<WorkerResult>();
        let work_receiver = Arc::new(work_receiver);
        let barrier = Arc::new(Barrier::new(num_threads + 1));
        let shutdown = Arc::new(AtomicBool::new(false));
        let work_counter = Arc::new(AtomicU64::new(0));
        let completed_tiles = Arc::new(RwLock::new(HashMap::new()));
        let recalibration_trigger = Arc::new((Mutex::new(false), Condvar::new()));
        let global_state = Arc::new(Mutex::new(0.0f32));

        let mut workers = Vec::new();

        for id in 0..num_threads {
            let work_receiver = Arc::clone(&work_receiver);
            let result_sender = result_sender.clone();
            let barrier = Arc::clone(&barrier);
            let shutdown = Arc::clone(&shutdown);
            let work_counter = Arc::clone(&work_counter);
            let recalib_trigger = Arc::clone(&recalibration_trigger);
            let global_state_ref = Arc::clone(&global_state);

            let handle = thread::Builder::new()
                .name(format!("worker-{}", id))
                .spawn(move || {
                    Self::worker_thread(
                        id,
                        work_receiver,
                        result_sender,
                        barrier,
                        shutdown,
                        work_counter,
                        recalib_trigger,
                        global_state_ref,
                    );
                })?;

            workers.push(handle);
        }

        Ok(ThreadPool {
            workers,
            sender: work_sender,
            result_receiver,
            barrier,
            shutdown,
            work_counter,
            completed_tiles,
            recalibration_trigger,
            global_state,
        })
    }

    fn worker_thread(
        id: usize,
        receiver: Arc<Receiver<WorkerMessage>>,
        sender: Sender<WorkerResult>,
        barrier: Arc<Barrier>,
        shutdown: Arc<AtomicBool>,
        work_counter: Arc<AtomicU64>,
        recalib_trigger: Arc<(Mutex<bool>, Condvar)>,
        global_state: Arc<Mutex<f32>>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            let msg = receiver.recv();

            match msg {
                Ok(WorkerMessage::ComputeTile(tile_work)) => {
                    let work_span = span!(
                        Level::INFO,
                        "compute_tile",
                        worker_id = id,
                        tile_id = tile_work.tile_id,
                        frame_id = tile_work.frame_id
                    );

                    if let Some(parent_id) = &tile_work.parent_span_id {
                        work_span.follows_from(parent_id.clone());
                    }

                    let _work_guard = work_span.enter();
                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // Check for system recalibration (non-blocking check)
                    if let Ok(trigger) = recalib_trigger.0.try_lock() {
                        if *trigger {
                            // System needs recalibration - update global state
                            drop(trigger);

                            // Update critical system parameters
                            if let Ok(mut state) = global_state.lock() {
                                // Perform complex state recalculation
                                for _ in 0..1000000 {
                                    *state = (*state * 1.1).sin();
                                }

                                // Track recalibration events for debugging
                                if tile_work.frame_id % 50 == 0 {
                                    let timestamp = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_millis();

                                    if let Ok(mut file) = std::fs::OpenOptions::new()
                                        .create(true)
                                        .append(true)
                                        .open("/tmp/fractal_perf.log")
                                    {
                                        use std::io::Write;
                                        let _ = writeln!(
                                            file,
                                            "{},tile_{},{:.6}",
                                            timestamp, tile_work.tile_id, state
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Compute tile parameters on CPU
                    let result = compute_tile_parameters(&tile_work, None);
                    sender.send(WorkerResult::TileComplete(result)).unwrap();
                }
                Ok(WorkerMessage::PreprocessData(data, parent_span_id)) => {
                    let work_span = span!(
                        Level::INFO,
                        "preprocess_data",
                        worker_id = id,
                        data_size = data.len()
                    );

                    // Establish follows_from relationship if parent span exists
                    if let Some(parent_id) = parent_span_id {
                        work_span.follows_from(parent_id);
                    }

                    let _work_guard = work_span.enter();

                    work_counter.fetch_add(1, Ordering::Relaxed);

                    // Check for system recalibration and update state
                    if let Ok(trigger) = recalib_trigger.0.try_lock() {
                        if *trigger {
                            drop(trigger);

                            // Update global processing state
                            if let Ok(mut state) = global_state.lock() {
                                // Complex state synchronization
                                for _ in 0..2000000 {
                                    *state = (*state * 1.1 + 0.1).sin();
                                }

                                // Log calibration metrics for performance analysis
                                let timestamp = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis();

                                if let Ok(mut file) = std::fs::OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open("/tmp/fractal_perf.log")
                                {
                                    use std::io::Write;
                                    let _ = writeln!(file, "{},{},{:.6}", timestamp, id, state);
                                }
                            }
                        }
                    }

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

    #[allow(dead_code)] // Used for periodic synchronization
    fn synchronize(&self, sync_id: u32) {
        for _ in 0..WORKER_THREADS {
            self.submit_work(WorkerMessage::Synchronize(sync_id));
        }
        self.barrier.wait();
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

    fn trigger_system_recalibration(&self) {
        // Periodic system recalibration for accuracy
        let (lock, cvar) = &*self.recalibration_trigger;
        if let Ok(mut trigger) = lock.lock() {
            *trigger = true;
            cvar.notify_all(); // Notify all workers of recalibration

            // Reset recalibration flag after processing window
            thread::Builder::new()
                .name("recalib-rst".to_string())
                .spawn({
                    let trigger_arc = Arc::clone(&self.recalibration_trigger);
                    move || {
                        thread::sleep(Duration::from_millis(500));
                        if let Ok(mut t) = trigger_arc.0.lock() {
                            *t = false;
                        }
                    }
                })
                .ok(); // Ignore spawn errors for cleanup thread
        }
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

fn schedule_tiles_for_frame(
    thread_pool: Arc<ThreadPool>,
    frame_id: u32,
    params: FractalParams,
    parent_span_id: Option<Id>,
) {
    // Create tiles for the frame
    let mut tile_id = 0;

    for tile_y in 0..TILES_Y {
        for tile_x in 0..TILES_X {
            let _x_start = tile_x * TILE_SIZE;
            let _y_start = tile_y * TILE_SIZE;

            let tile_work = TileWork {
                tile_id,
                frame_id,
                params,
                parent_span_id: parent_span_id.clone(),
            };

            thread_pool.submit_work(WorkerMessage::ComputeTile(tile_work));
            tile_id += 1;
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
        let thread_pool = Arc::new(ThreadPool::new(WORKER_THREADS)?);

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

                    vec3 hsv_to_rgb(float h, float s, float v) {
                        h = fract(h);
                        float i = floor(h * 6.0);
                        float f = h * 6.0 - i;
                        float p = v * (1.0 - s);
                        float q = v * (1.0 - s * f);
                        float t = v * (1.0 - s * (1.0 - f));
                        
                        int idx = int(mod(i, 6.0));
                        vec3 rgb;
                        if (idx == 0) rgb = vec3(v, t, p);
                        else if (idx == 1) rgb = vec3(q, v, p);
                        else if (idx == 2) rgb = vec3(p, v, t);
                        else if (idx == 3) rgb = vec3(p, q, v);
                        else if (idx == 4) rgb = vec3(t, p, v);
                        else rgb = vec3(v, p, q);
                        
                        return rgb;
                    }

                    void main() {
                        ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

                        if (pixel_coords.x >= params.width || pixel_coords.y >= params.height) {
                            return;
                        }

                        // Get CPU-computed influence/parameters (small amount of data)
                        vec4 cpu_influence = imageLoad(cpu_data, pixel_coords / 64); // Sample at tile level
                        
                        // Convert to complex plane coordinates
                        float aspect_ratio = float(params.width) / float(params.height);
                        float real = (float(pixel_coords.x) / float(params.width) - 0.5) * 4.0 / params.zoom * aspect_ratio + params.center_x;
                        float imag = (float(pixel_coords.y) / float(params.height) - 0.5) * 4.0 / params.zoom + params.center_y;
                        
                        // Apply CPU influence to starting position
                        real += cpu_influence.r * 0.1;
                        imag += cpu_influence.g * 0.1;
                        
                        // Julia set parameters with time variation - slower animation
                        float c_real = params.julia_c_real + 0.1 * sin(params.time * 0.3);
                        float c_imag = params.julia_c_imag + 0.1 * cos(params.time * 0.2);
                        
                        // Julia iteration
                        float z_real = real;
                        float z_imag = imag;
                        uint iteration = 0;
                        float trap_min = 1000.0;
                        
                        for (uint i = 0; i < params.max_iterations; i++) {
                            float z_real_squared = z_real * z_real;
                            float z_imag_squared = z_imag * z_imag;
                            
                            // Orbit trap for additional coloring
                            float dist = sqrt((z_real - 0.5) * (z_real - 0.5) + z_imag * z_imag);
                            trap_min = min(trap_min, dist);
                            
                            if (z_real_squared + z_imag_squared > 4.0) {
                                float smooth_iter = float(i) + 1.0 - log(log(sqrt(z_real_squared + z_imag_squared))) / log(2.0);
                                
                                // Gentle color scheme - no bright flashes
                                float normalized = smooth_iter / float(params.max_iterations);
                                float hue = normalized * 2.0 + params.time * 0.05;
                                float saturation = 0.6 + 0.3 * sin(normalized * 3.14159);
                                float value = 0.2 + normalized * 0.4; // Clamped to max 0.6 brightness
                                vec3 rgb = hsv_to_rgb(hue, saturation, value);
                                imageStore(img, pixel_coords, vec4(rgb, 1.0));
                                return;
                            }
                            
                            float new_real = z_real_squared - z_imag_squared + c_real;
                            float new_imag = 2.0 * z_real * z_imag + c_imag;
                            
                            z_real = new_real;
                            z_imag = new_imag;
                            iteration = i;
                        }
                        
                        // Very dark background - no flashing
                        vec3 color = vec3(0.01, 0.01, 0.05);
                        imageStore(img, pixel_coords, vec4(color, 1.0));
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
        // Much smaller image for tile-level influence data
        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                extent: [TILES_X, TILES_Y, 1], // One pixel per tile
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

        // Periodic system recalibration for rendering accuracy
        if frame_id % 240 == 180 {
            // Every ~4 seconds at 60fps
            self.thread_pool.trigger_system_recalibration();
        }

        // Schedule tile-based parameter computation (much lighter than fractal)
        {
            let tile_schedule_span = span!(Level::INFO, "schedule_tiles", frame_id = frame_id);
            let _schedule_guard = tile_schedule_span.enter();
            schedule_tiles_for_frame(
                Arc::clone(&self.thread_pool),
                frame_id,
                params,
                tile_schedule_span.id(),
            );
        }

        // Wait for parameter tiles (now much faster)
        let total_tiles = TILES_X * TILES_Y;
        let completed_tiles = self.thread_pool.wait_for_frame_tiles(frame_id, total_tiles);

        // Create small influence map from tile parameters
        let influence_data = self.create_influence_map(completed_tiles);

        // Upload small CPU-computed influence data to GPU
        self.upload_cpu_data_to_gpu(&influence_data)?;

        // Periodic data validation and preprocessing
        if frame_id % 120 == 60 {
            // Every 2 seconds
            // Submit validation work for data integrity
            for _ in 0..WORKER_THREADS {
                let validation_data = vec![frame_id as f32; 1000];
                self.thread_pool
                    .submit_work(WorkerMessage::PreprocessData(validation_data, None));
            }
        }

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

    fn create_influence_map(&self, tiles: HashMap<u32, TileResult>) -> Vec<[u8; 4]> {
        // Create a small influence map (one value per tile)
        let influence_size = (TILES_X * TILES_Y) as usize;
        let mut influence_map = vec![[128u8; 4]; influence_size];

        for tile_result in tiles.values() {
            if let Some(first_pixel) = tile_result.pixel_data.first() {
                let tile_idx = tile_result.tile_id as usize;
                if tile_idx < influence_map.len() {
                    influence_map[tile_idx] = *first_pixel;
                }
            }
        }

        influence_map
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

    // Removed - no longer needed since GPU does the rendering

    // Removed - no longer needed since GPU does the rendering directly
}

fn setup_tracing() {
    let perfetto_layer = tracing_perfetto::PerfettoLayer::new(
        std::fs::File::create("trace.perfetto-trace").unwrap(),
    );

    // Enable both perfetto and chrome layers
    tracing_subscriber::registry().with(perfetto_layer).init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set main thread name (cross-platform)
    #[cfg(target_os = "linux")]
    unsafe {
        libc::pthread_setname_np(libc::pthread_self(), b"main\0".as_ptr() as *const i8);
    }
    #[cfg(target_os = "macos")]
    unsafe {
        libc::pthread_setname_np(b"main\0".as_ptr() as *const i8);
    }
    setup_tracing();

    let event_loop = EventLoop::new();
    let (mut renderer, window) = VulkanRenderer::new(&event_loop)?;

    let mut frame_count = 0u32;
    let start_time = std::time::Instant::now();

    // FPS tracking variables
    let mut fps_timer = std::time::Instant::now();
    let mut fps_frame_count = 0u32;
    let mut frame_times = VecDeque::with_capacity(60);
    let mut last_frame_time = std::time::Instant::now();

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
                let now = std::time::Instant::now();
                let frame_time = now.duration_since(last_frame_time);
                last_frame_time = now;

                // Track frame times for averaging
                frame_times.push_back(frame_time.as_secs_f32() * 1000.0);
                if frame_times.len() > 60 {
                    frame_times.pop_front();
                }

                // Update FPS counter every second
                fps_frame_count += 1;
                if fps_timer.elapsed().as_secs_f32() >= 1.0 {
                    let current_fps = fps_frame_count as f32 / fps_timer.elapsed().as_secs_f32();
                    fps_frame_count = 0;
                    fps_timer = std::time::Instant::now();

                    // Calculate average frame time
                    let avg_frame_time = if !frame_times.is_empty() {
                        frame_times.iter().sum::<f32>() / frame_times.len() as f32
                    } else {
                        0.0
                    };

                    // Update window title with FPS
                    window.set_title(&format!(
                        "Multi-threaded GPU Fractal Renderer - FPS: {:.1} | Frame Time: {:.2}ms",
                        current_fps, avg_frame_time
                    ));
                }

                let elapsed = start_time.elapsed().as_secs_f32();

                // Adaptive global animation speed based on Julia set parameters
                let base_c_real = -0.7;
                let base_c_imag = 0.0;
                let current_c_real = base_c_real + 0.1 * (elapsed * 0.3).sin();
                let current_c_imag = base_c_imag + 0.1 * (elapsed * 0.2).cos();

                // Calculate distance from stable Julia set parameters
                let stability_factor =
                    (current_c_real * current_c_real + current_c_imag * current_c_imag).sqrt();
                let animation_speed = if stability_factor < 0.5 {
                    0.3 // Slow in stable regions
                } else {
                    1.0 + stability_factor * 2.0 // Speed up in chaotic regions
                };

                let adaptive_time = elapsed * animation_speed;

                let mut params = FractalParams::default();
                params.time = adaptive_time;
                params.zoom = 1.0;
                params.center_x = 0.0;
                params.center_y = 0.0;

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
