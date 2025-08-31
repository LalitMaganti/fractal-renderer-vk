use std::sync::{Arc, Barrier, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crossbeam_channel::{unbounded, Sender, Receiver};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::VecDeque;

use tracing::{span, Level, Span};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image};

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const ITERATIONS: u32 = 256;
const WORKER_THREADS: usize = 8;
const COMPUTE_THREADS: usize = 4;

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
    PreprocessData(Vec<f32>, Option<Span>),
    PostprocessData(Vec<u8>, Option<Span>),
    Synchronize(u32),
    Shutdown,
}

#[allow(dead_code)]
#[derive(Debug)]
enum WorkerResult {
    FrameReady(u32, FractalParams),
    DataProcessed(Vec<f32>),
    ImageProcessed(Vec<u8>),
    SyncComplete(u32),
}


struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: Sender<WorkerMessage>,
    #[allow(dead_code)]
    result_receiver: Receiver<WorkerResult>,
    barrier: Arc<Barrier>,
    shutdown: Arc<AtomicBool>,
    #[allow(dead_code)]
    work_counter: Arc<AtomicU64>,
}

impl ThreadPool {
    fn new(num_threads: usize) -> Self {
        let (work_sender, work_receiver) = unbounded::<WorkerMessage>();
        let (result_sender, result_receiver) = unbounded::<WorkerResult>();
        let work_receiver = Arc::new(work_receiver);
        let barrier = Arc::new(Barrier::new(num_threads + 1));
        let shutdown = Arc::new(AtomicBool::new(false));
        let work_counter = Arc::new(AtomicU64::new(0));

        let mut workers = Vec::new();

        for id in 0..num_threads {
            let work_receiver = Arc::clone(&work_receiver);
            let result_sender = result_sender.clone();
            let barrier = Arc::clone(&barrier);
            let shutdown = Arc::clone(&shutdown);
            let work_counter = Arc::clone(&work_counter);

            let handle = thread::spawn(move || {
                Self::worker_thread(id, work_receiver, result_sender, barrier, shutdown, work_counter);
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
                    let work_span = span!(Level::INFO, "compute_frame", 
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

                    sender.send(WorkerResult::FrameReady(frame_id, params)).unwrap();
                }
                Ok(WorkerMessage::PreprocessData(data, parent_span)) => {
                    let work_span = span!(Level::INFO, "preprocess_data", 
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
                    let processed: Vec<f32> = data.iter()
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
                    let work_span = span!(Level::INFO, "postprocess_data", 
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
                    let processed: Vec<u8> = data.iter()
                        .map(|&pixel| {
                            let mut p = pixel as f32 / 255.0;
                            p = p.powf(2.2); // Gamma correction
                            (p * 255.0) as u8
                        })
                        .collect();

                    sender.send(WorkerResult::ImageProcessed(processed)).unwrap();
                }
                Ok(WorkerMessage::Synchronize(sync_id)) => {
                    let sync_span = span!(Level::INFO, "synchronize", 
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
                let schedule_span = span!(Level::INFO, "schedule_frame",
                    compute_id = id,
                    frame_id = frame_id
                );
                let _schedule_guard = schedule_span.enter();

                // Submit preprocessing work to thread pool with parent span
                let preprocess_data = vec![params.time; 1000];
                thread_pool.submit_work(WorkerMessage::PreprocessData(
                    preprocess_data, 
                    parent_span.clone()
                ));

                // Submit frame computation with parent span
                thread_pool.submit_work(WorkerMessage::ComputeFrame(
                    frame_id, 
                    params, 
                    parent_span.clone()
                ));

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
    #[allow(dead_code)]
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    compute_pipeline: Arc<ComputePipeline>,
    image: Arc<ImageView>,
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
        let window = Arc::new(WindowBuilder::new()
            .with_title("Multi-threaded GPU Fractal Renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .build(event_loop)?);

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

        // Create output image with RGBA8 format for compute shader
        let image = Self::create_output_image(&memory_allocator)?;

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
            image,
            params_buffer,
            thread_pool,
            scheduler,
            surface,
            swapchain,
            swapchain_images,
        };

        Ok((renderer, window.clone()))
    }

    fn create_instance(library: &Arc<VulkanLibrary>, required_extensions: &vulkano::instance::InstanceExtensions) -> Result<Arc<Instance>, Box<dyn std::error::Error>> {

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

    fn select_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface>) -> Result<Arc<PhysicalDevice>, Box<dyn std::error::Error>> {

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
            .min_by_key(|p| {
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
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
                queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
                    && physical_device.surface_support(i as u32, surface).unwrap_or(false)
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

                    layout(set = 0, binding = 1) uniform FractalParams {
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

                    vec3 hsv2rgb(vec3 c) {
                        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                    }

                    void main() {
                        ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

                        if (pixel_coords.x >= params.width || pixel_coords.y >= params.height) {
                            return;
                        }

                        float aspect_ratio = float(params.width) / float(params.height);

                        // Convert pixel coordinates to complex plane
                        float real = (float(pixel_coords.x) / float(params.width) - 0.5) * 4.0 / params.zoom * aspect_ratio + params.center_x;
                        float imag = (float(pixel_coords.y) / float(params.height) - 0.5) * 4.0 / params.zoom + params.center_y;

                        // Julia set iteration with more complex calculations
                        float z_real = real;
                        float z_imag = imag;

                        // Animate the Julia constant
                        float c_real = params.julia_c_real + 0.1 * sin(params.time);
                        float c_imag = params.julia_c_imag + 0.1 * cos(params.time * 0.7);

                        uint iteration = 0;
                        float smooth_iter = 0.0;
                        float trap_min = 1000.0;

                        for (iteration = 0; iteration < params.max_iterations; iteration++) {
                            float z_real_squared = z_real * z_real;
                            float z_imag_squared = z_imag * z_imag;

                            // Orbit trap for additional coloring
                            float dist = sqrt((z_real - 0.5) * (z_real - 0.5) + z_imag * z_imag);
                            trap_min = min(trap_min, dist);

                            if (z_real_squared + z_imag_squared > 4.0) {
                                smooth_iter = float(iteration) + 1.0 - log(log(sqrt(z_real_squared + z_imag_squared))) / log(2.0);
                                break;
                            }

                            float new_real = z_real_squared - z_imag_squared + c_real;
                            float new_imag = 2.0 * z_real * z_imag + c_imag;

                            z_real = new_real;
                            z_imag = new_imag;
                        }

                        vec3 color;
                        if (iteration == params.max_iterations) {
                            float interior = trap_min * 3.0;
                            color = vec3(interior * 0.1, interior * 0.2, interior * 0.3);
                        } else {
                            float hue = smooth_iter / float(params.max_iterations) * 3.0 + params.time * 0.1 + trap_min;
                            float saturation = 0.8 - trap_min * 0.3;
                            float value = 1.0 - pow(smooth_iter / float(params.max_iterations), 0.5);
                            color = hsv2rgb(vec3(hue, saturation, value));
                        }

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
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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

        let surface_capabilities = device.physical_device().surface_capabilities(surface, Default::default())?;
        let image_format = device.physical_device()
            .surface_formats(surface, Default::default())?[0].0;

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: [WIDTH, HEIGHT],
                image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT | vulkano::image::ImageUsage::TRANSFER_DST,
                composite_alpha: surface_capabilities.supported_composite_alpha.into_iter().next().unwrap(),
                ..Default::default()
            },
        )?;

        let swapchain_images = images
            .into_iter()
            .map(|image| ImageView::new_default(image))
            .collect::<Result<Vec<_>, _>>()?;

        Ok((swapchain, swapchain_images))
    }

    pub fn render_frame(&mut self, frame_id: u32, params: FractalParams) -> Result<(), Box<dyn std::error::Error>> {
        let span = span!(Level::INFO, "render_frame", frame_id = frame_id);
        let _enter = span.enter();

        // Schedule preprocessing work on thread pool
        self.scheduler.schedule_frame(frame_id, params, Some(span.clone()));

        // Simulate some post-processing work
        let dummy_image_data = vec![128u8; 1000];
        self.thread_pool.submit_work(WorkerMessage::PostprocessData(dummy_image_data, None));

        // Update parameters
        {
            let mut content = self.params_buffer.write()?;
            *content = params;
        }

        // Acquire next swapchain image
        let (image_index, _suboptimal, acquire_future) = acquire_next_image(self.swapchain.clone(), None)?;
        let swapchain_image = &self.swapchain_images[image_index as usize];

        // Create descriptor set
        let layout = self.compute_pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.image.clone()),
                WriteDescriptorSet::buffer(1, self.params_buffer.clone()),
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
                self.image.image().clone(),
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
                self.thread_pool.submit_work(WorkerMessage::PreprocessData(work_data, None));
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
    let perfetto_layer = tracing_perfetto::PerfettoLayer::new(std::fs::File::create("trace.perfetto-trace").unwrap());
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().file("trace-chrome.json").build();

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
            }
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
