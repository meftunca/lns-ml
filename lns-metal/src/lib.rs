use half::f16 as F16;
use lazy_static::lazy_static;
use lns_core::format::QuantType;
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize, ResourceRef,
};
use std::collections::HashMap;
use std::os::raw::c_void;

pub struct MetalBuffer(pub Buffer, pub QuantType);

pub struct GemvBatchJob<'a> {
    pub weights: &'a MetalBuffer,
    pub y: &'a Buffer,
    pub out_dim: usize,
    pub in_dim: usize,
    pub qt: QuantType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ComputeBackend {
    Cpu,
    Metal,
}

#[derive(Debug, Clone)]
pub struct DecodeBenchResult {
    pub elapsed_secs: f64,
    pub weights_processed: usize,
    pub checksum: f32,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub zero_fraction: f64,
    pub code_entropy_bits: f64,
    pub roundtrip_rmse: f64,
    pub roundtrip_snr_db: f64,
    pub n_weights: usize,
}

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub q4l_decode_pipeline: Option<ComputePipelineState>,
    pub q2l_decode_pipeline: Option<ComputePipelineState>,
    pub q8l_decode_pipeline: Option<ComputePipelineState>,
    pub rmsnorm_pipeline: Option<ComputePipelineState>,
    pub rmsnorm_rows_pipeline: Option<ComputePipelineState>,
    pub rope_pipeline: Option<ComputePipelineState>,
    pub rope_with_freqs_pipeline: Option<ComputePipelineState>,
    pub flash_attention_pipeline: Option<ComputePipelineState>,
    pub flash_attention_q8a_pipeline: Option<ComputePipelineState>,
    pub flash_attention_paged_q8a_pipeline: Option<ComputePipelineState>,
    pub f16_gemv_pipeline: Option<ComputePipelineState>,
    pub f16_gemv_multirow_pipeline: Option<ComputePipelineState>,
    pub swiglu_pipeline: Option<ComputePipelineState>,
    pub linear_attn_conv_pipeline: Option<ComputePipelineState>,
    pub linear_attn_recurrent_pipeline: Option<ComputePipelineState>,
    pub split_gated_query_pipeline: Option<ComputePipelineState>,
    pub mul_inplace_pipeline: Option<ComputePipelineState>,
    pub add_inplace_pipeline: Option<ComputePipelineState>,
    pub q4l_gemv_accumulate_pipeline: Option<ComputePipelineState>,
    pub q4hq_gemv_accumulate_pipeline: Option<ComputePipelineState>,
    pub gemv_pipelines: HashMap<QuantType, ComputePipelineState>,
    // Multi-row (4 rows/TG) variants — higher bandwidth utilization
    pub q4l_gemv_multirow_pipeline: Option<ComputePipelineState>,
    pub q4l_gemv_accumulate_multirow_pipeline: Option<ComputePipelineState>,
    pub q4hq_gemv_multirow_pipeline: Option<ComputePipelineState>,
    pub q4hq_gemv_accumulate_multirow_pipeline: Option<ComputePipelineState>,
    // Fused RMSNorm + GEMV (eliminates norm→GEMV barrier)
    pub q4l_rmsnorm_gemv_multirow_pipeline: Option<ComputePipelineState>,
    pub q4l_rmsnorm_gemv_accumulate_multirow_pipeline: Option<ComputePipelineState>,
    pub q4hq_rmsnorm_gemv_multirow_pipeline: Option<ComputePipelineState>,
    // Fused W1+W3+SwiGLU (eliminates swiglu dispatch and one GEMV dispatch)
    pub q4l_w1w3_swiglu_multirow_pipeline: Option<ComputePipelineState>,
    pub q4hq_w1w3_swiglu_multirow_pipeline: Option<ComputePipelineState>,
    // GPU argmax for greedy decode
    pub argmax_pass1_pipeline: Option<ComputePipelineState>,
    pub argmax_pass2_pipeline: Option<ComputePipelineState>,
    // GPU argmax scratch buffer  (ceil(vocab/256)*2 floats)
    pub argmax_partials_buf: Option<Buffer>,
    pub argmax_result_buf: Option<Buffer>,
}

lazy_static! {
    static ref METAL_CONTEXT: Option<MetalContext> = {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("Metal initialization failed: {}", e);
                None
            }
        }
    };
}

pub fn get_metal_context() -> Option<&'static MetalContext> {
    METAL_CONTEXT.as_ref()
}

impl MetalContext {
    fn create_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let kernel = library
            .get_function(name, None)
            .map_err(|e| format!("Kernel '{}' not found: {}", name, e))?;
        device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Pipeline '{}' creation failed: {}", name, e))
    }

    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let source = lns_decode_kernel_source();
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|e| format!("Shader compilation failed: {}", e))?;

        let q4l_decode_pipeline =
            Self::create_pipeline(&device, &library, "q4l_decode_kernel_v2").ok();
        let q2l_decode_pipeline =
            Self::create_pipeline(&device, &library, "q2l_decode_kernel_v2").ok();
        let q8l_decode_pipeline =
            Self::create_pipeline(&device, &library, "q8l_decode_kernel_v2").ok();
        let rmsnorm_pipeline = Self::create_pipeline(&device, &library, "rmsnorm_kernel").ok();
        let rmsnorm_rows_pipeline =
            Self::create_pipeline(&device, &library, "rmsnorm_rows_kernel").ok();
        let rope_pipeline = Self::create_pipeline(&device, &library, "rope_optimized").ok();
        let rope_with_freqs_pipeline =
            Self::create_pipeline(&device, &library, "rope_with_freqs").ok();
        let flash_attention_pipeline =
            Self::create_pipeline(&device, &library, "flash_attention_decode").ok();
        let flash_attention_q8a_pipeline =
            Self::create_pipeline(&device, &library, "flash_attention_q8a").ok();
        let flash_attention_paged_q8a_pipeline =
            Self::create_pipeline(&device, &library, "flash_attention_paged_q8a").ok();
        let f16_gemv_pipeline = Self::create_pipeline(&device, &library, "f16_gemv_kernel").ok();
        let f16_gemv_multirow_pipeline =
            Self::create_pipeline(&device, &library, "f16_gemv_multirow").ok();
        let swiglu_pipeline = Self::create_pipeline(&device, &library, "swiglu_kernel").ok();
        let linear_attn_conv_pipeline =
            Self::create_pipeline(&device, &library, "linear_attn_conv1d").ok();
        let linear_attn_recurrent_pipeline =
            Self::create_pipeline(&device, &library, "linear_attn_recurrent").ok();
        let split_gated_query_pipeline =
            Self::create_pipeline(&device, &library, "split_gated_query_kernel").ok();
        let mul_inplace_pipeline =
            Self::create_pipeline(&device, &library, "mul_inplace_kernel").ok();
        let add_inplace_pipeline =
            Self::create_pipeline(&device, &library, "add_inplace_kernel").ok();
        let q4l_gemv_accumulate_pipeline =
            Self::create_pipeline(&device, &library, "q4l_gemv_accumulate_optimized").ok();
        let q4hq_gemv_accumulate_pipeline =
            Self::create_pipeline(&device, &library, "q4hq_gemv_accumulate_optimized").ok();

        let mut gemv_pipelines = HashMap::new();
        if let Ok(p) = Self::create_pipeline(&device, &library, "q4l_gemv_optimized") {
            gemv_pipelines.insert(QuantType::Q4L, p);
        }
        if let Ok(p) = Self::create_pipeline(&device, &library, "q4hq_gemv_optimized") {
            gemv_pipelines.insert(QuantType::Q4HQ, p);
        }
        if let Ok(p) = Self::create_pipeline(&device, &library, "q2l_gemv_kernel") {
            gemv_pipelines.insert(QuantType::Q2L, p);
        }
        if let Ok(p) = Self::create_pipeline(&device, &library, "q8l_gemv_kernel") {
            gemv_pipelines.insert(QuantType::Q8L, p);
        }

        let q4l_gemv_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4l_gemv_multirow").ok();
        let q4l_gemv_accumulate_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4l_gemv_accumulate_multirow").ok();
        let q4hq_gemv_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4hq_gemv_multirow").ok();
        let q4hq_gemv_accumulate_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4hq_gemv_accumulate_multirow").ok();
        let q4l_rmsnorm_gemv_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4l_rmsnorm_gemv_multirow").ok();
        let q4l_rmsnorm_gemv_accumulate_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4l_rmsnorm_gemv_accumulate_multirow").ok();
        let q4hq_rmsnorm_gemv_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4hq_rmsnorm_gemv_multirow").ok();
        let q4l_w1w3_swiglu_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4l_w1w3_swiglu_multirow").ok();
        let q4hq_w1w3_swiglu_multirow_pipeline =
            Self::create_pipeline(&device, &library, "q4hq_w1w3_swiglu_multirow").ok();
        let argmax_pass1_pipeline = Self::create_pipeline(&device, &library, "argmax_pass1").ok();
        let argmax_pass2_pipeline = Self::create_pipeline(&device, &library, "argmax_pass2").ok();
        // Preallocate argmax scratch: enough for vocab_size=262144 (256Ki)
        let max_vocab = 262144usize;
        let n_parts = (max_vocab + 255) / 256;
        let argmax_partials_buf = Some(device.new_buffer(
            (n_parts * 2 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let argmax_result_buf = Some(device.new_buffer(4, MTLResourceOptions::StorageModeShared));

        Ok(Self {
            device,
            queue,
            q4l_decode_pipeline,
            q2l_decode_pipeline,
            q8l_decode_pipeline,
            rmsnorm_pipeline,
            rmsnorm_rows_pipeline,
            rope_pipeline,
            rope_with_freqs_pipeline,
            flash_attention_pipeline,
            flash_attention_q8a_pipeline,
            flash_attention_paged_q8a_pipeline,
            f16_gemv_pipeline,
            f16_gemv_multirow_pipeline,
            swiglu_pipeline,
            linear_attn_conv_pipeline,
            linear_attn_recurrent_pipeline,
            split_gated_query_pipeline,
            mul_inplace_pipeline,
            add_inplace_pipeline,
            q4l_gemv_accumulate_pipeline,
            q4hq_gemv_accumulate_pipeline,
            gemv_pipelines,
            q4l_gemv_multirow_pipeline,
            q4l_gemv_accumulate_multirow_pipeline,
            q4hq_gemv_multirow_pipeline,
            q4hq_gemv_accumulate_multirow_pipeline,
            q4l_rmsnorm_gemv_multirow_pipeline,
            q4l_rmsnorm_gemv_accumulate_multirow_pipeline,
            q4hq_rmsnorm_gemv_multirow_pipeline,
            q4l_w1w3_swiglu_multirow_pipeline,
            q4hq_w1w3_swiglu_multirow_pipeline,
            argmax_pass1_pipeline,
            argmax_pass2_pipeline,
            argmax_partials_buf,
            argmax_result_buf,
        })
    }

    fn gemv_multirow_pipeline_for(&self, quant_type: QuantType) -> Option<&ComputePipelineState> {
        match quant_type {
            QuantType::Q4L => self.q4l_gemv_multirow_pipeline.as_ref(),
            QuantType::Q4HQ => self.q4hq_gemv_multirow_pipeline.as_ref(),
            _ => None,
        }
    }

    fn gemv_accumulate_pipeline_for(&self, quant_type: QuantType) -> Option<&ComputePipelineState> {
        match quant_type {
            QuantType::Q4L => self.q4l_gemv_accumulate_pipeline.as_ref(),
            QuantType::Q4HQ => self.q4hq_gemv_accumulate_pipeline.as_ref(),
            _ => None,
        }
    }

    fn gemv_accumulate_multirow_pipeline_for(
        &self,
        quant_type: QuantType,
    ) -> Option<&ComputePipelineState> {
        match quant_type {
            QuantType::Q4L => self.q4l_gemv_accumulate_multirow_pipeline.as_ref(),
            QuantType::Q4HQ => self.q4hq_gemv_accumulate_multirow_pipeline.as_ref(),
            _ => None,
        }
    }

    pub fn prepare_f16_dense_tensor(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn prepare_f32_dense_tensor(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (std::mem::size_of_val(data)) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn encode_rmsnorm_buffer(
        &self,
        cb: &CommandBufferRef,
        x: &Buffer,
        w: &Buffer,
        out: &Buffer,
        dim: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .rmsnorm_pipeline
            .as_ref()
            .ok_or("No RMSNorm pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(out), 0);
        let dim_u32 = dim as u32;
        let zero_centered_u32 = if zero_centered { 1u32 } else { 0u32 };
        encoder.set_bytes(3, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        encoder.set_bytes(5, 4, &zero_centered_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_rmsnorm_rows_buffer(
        &self,
        cb: &CommandBufferRef,
        x: &Buffer,
        w: &Buffer,
        row_dim: usize,
        rows: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .rmsnorm_rows_pipeline
            .as_ref()
            .ok_or("No row RMSNorm pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        let row_dim_u32 = row_dim as u32;
        let rows_u32 = rows as u32;
        let zero_centered_u32 = if zero_centered { 1u32 } else { 0u32 };
        encoder.set_bytes(2, 4, &row_dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &rows_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        encoder.set_bytes(5, 4, &zero_centered_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_gemv_batch_cached(
        &self,
        cb: &CommandBufferRef,
        x: &Buffer,
        jobs: &[GemvBatchJob<'_>],
    ) -> Result<(), String> {
        if jobs.is_empty() {
            return Ok(());
        }

        // Use a single compute encoder with multiple dispatches.
        // On Apple Silicon, independent dispatches within one encoder can run
        // concurrently on the GPU — much faster than separate encoders which
        // force serialization and pay alloc+end overhead per GEMV.
        let first_quant_type = jobs[0].weights.1;
        let first_pipeline = self
            .gemv_pipelines
            .get(&first_quant_type)
            .ok_or("Pipeline missing")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(first_pipeline);
        encoder.set_buffer(1, Some(x), 0);

        for job in jobs {
            // Switch pipeline only if the quant type changes (rare in practice).
            let quant_type = job.weights.1;
            if quant_type != first_quant_type {
                let pipeline = self
                    .gemv_pipelines
                    .get(&quant_type)
                    .ok_or("Pipeline missing")?;
                encoder.set_compute_pipeline_state(pipeline);
            }
            encoder.set_buffer(0, Some(&job.weights.0), 0);
            encoder.set_buffer(2, Some(job.y), 0);
            let ncols = job.in_dim as u32;
            encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(
                MTLSize::new(job.out_dim as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_gemv_accumulate_cached(
        &self,
        cb: &CommandBufferRef,
        weights: &MetalBuffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .gemv_accumulate_pipeline_for(weights.1)
            .ok_or("Accumulate pipeline missing")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weights.0), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(y), 0);
        let ncols = in_dim as u32;
        encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    // ── Multi-row GEMV (4 rows/threadgroup) ──────────────────────────────────
    // dispatch: ceil(out_dim/4) threadgroups × 256 threads

    pub fn encode_gemv_batch_multirow(
        &self,
        cb: &CommandBufferRef,
        x: &Buffer,
        jobs: &[GemvBatchJob<'_>],
    ) -> Result<(), String> {
        for job in jobs {
            let quant_type = job.weights.1;
            let Some(pipeline) = self.gemv_multirow_pipeline_for(quant_type) else {
                let fb = self
                    .gemv_pipelines
                    .get(&quant_type)
                    .ok_or("Pipeline missing")?;
                let encoder = cb.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(fb);
                encoder.set_buffer(0, Some(&job.weights.0), 0);
                encoder.set_buffer(1, Some(x), 0);
                encoder.set_buffer(2, Some(job.y), 0);
                let ncols = job.in_dim as u32;
                encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
                encoder.dispatch_thread_groups(
                    MTLSize::new(job.out_dim as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                encoder.end_encoding();
                continue;
            };
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&job.weights.0), 0);
            encoder.set_buffer(1, Some(x), 0);
            encoder.set_buffer(2, Some(job.y), 0);
            let ncols = job.in_dim as u32;
            let nrows = job.out_dim as u32;
            let n_tg = ((job.out_dim + 3) / 4) as u64;
            encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &nrows as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
            encoder.end_encoding();
        }
        Ok(())
    }

    pub fn encode_gemv_accumulate_multirow(
        &self,
        cb: &CommandBufferRef,
        weights: &MetalBuffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        let pipeline = match self.gemv_accumulate_multirow_pipeline_for(weights.1) {
            Some(p) => p,
            None => return self.encode_gemv_accumulate_cached(cb, weights, x, y, out_dim, in_dim),
        };
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weights.0), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(y), 0);
        let ncols = in_dim as u32;
        let nrows = out_dim as u32;
        let n_tg = ((out_dim + 3) / 4) as u64;
        encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &nrows as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    // ── Fused RMSNorm + multirow GEMV ─────────────────────────────────────────

    pub fn encode_rmsnorm_gemv_multirow(
        &self,
        cb: &CommandBufferRef,
        x: &Buffer,
        norm_w: &Buffer,
        weights: &MetalBuffer,
        y: &Buffer,
        dim: usize,
        out_rows: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .q4l_rmsnorm_gemv_multirow_pipeline
            .as_ref()
            .ok_or("RMSNorm+GEMV multirow pipeline missing")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(norm_w), 0);
        encoder.set_buffer(2, Some(&weights.0), 0);
        encoder.set_buffer(3, Some(y), 0);
        let u_dim = dim as u32;
        let u_out_rows = out_rows as u32;
        let u_zc = if zero_centered { 1u32 } else { 0u32 };
        encoder.set_bytes(4, 4, &u_dim as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &u_out_rows as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const c_void);
        encoder.set_bytes(7, 4, &u_zc as *const u32 as *const c_void);
        let n_tg = ((out_rows + 3) / 4) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_rmsnorm_gemv_accumulate_multirow(
        &self,
        cb: &CommandBufferRef,
        x: &Buffer,
        norm_w: &Buffer,
        weights: &MetalBuffer,
        y: &Buffer,
        dim: usize,
        out_rows: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .q4l_rmsnorm_gemv_accumulate_multirow_pipeline
            .as_ref()
            .ok_or("RMSNorm+GEMV accumulate multirow pipeline missing")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(norm_w), 0);
        encoder.set_buffer(2, Some(&weights.0), 0);
        encoder.set_buffer(3, Some(y), 0);
        let u_dim = dim as u32;
        let u_out_rows = out_rows as u32;
        let u_zc = if zero_centered { 1u32 } else { 0u32 };
        encoder.set_bytes(4, 4, &u_dim as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &u_out_rows as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const c_void);
        encoder.set_bytes(7, 4, &u_zc as *const u32 as *const c_void);
        let n_tg = ((out_rows + 3) / 4) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    // ── GPU argmax (greedy decode) ────────────────────────────────────────────
    // Returns the argmax token id directly without reading back all logits.
    // Requires the logits to already be in `logits_buf` (committed to device memory
    // prior to calling this inside a new CB, or encoded in same CB before commit).

    pub fn dispatch_argmax(&self, logits_buf: &Buffer, vocab_size: usize) -> Result<u32, String> {
        let p1 = self
            .argmax_pass1_pipeline
            .as_ref()
            .ok_or("argmax_pass1 pipeline missing")?;
        let p2 = self
            .argmax_pass2_pipeline
            .as_ref()
            .ok_or("argmax_pass2 pipeline missing")?;
        let partials = self
            .argmax_partials_buf
            .as_ref()
            .ok_or("argmax partials buf missing")?;
        let result = self
            .argmax_result_buf
            .as_ref()
            .ok_or("argmax result buf missing")?;

        let n_parts = ((vocab_size + 255) / 256) as u32;
        let cb = self.queue.new_command_buffer();

        {
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(p1);
            encoder.set_buffer(0, Some(logits_buf), 0);
            encoder.set_buffer(1, Some(partials), 0);
            let n = vocab_size as u32;
            encoder.set_bytes(2, 4, &n as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(
                MTLSize::new(n_parts as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
        }
        {
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(p2);
            encoder.set_buffer(0, Some(partials), 0);
            encoder.set_buffer(1, Some(result), 0);
            encoder.set_bytes(2, 4, &n_parts as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            encoder.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();

        let token_id = unsafe { *(result.contents() as *const u32) };
        Ok(token_id)
    }

    // ── GPU argmax encodeable variant (within existing CB) ───────────────────
    // Use this when the logits GEMV is encoded in the same CB.
    // The argmax runs immediately after logits, in two chained encoders.
    // Result readable after cb.wait_until_completed().

    pub fn encode_argmax(
        &self,
        cb: &CommandBufferRef,
        logits_buf: &Buffer,
        vocab_size: usize,
    ) -> Result<(), String> {
        let p1 = self
            .argmax_pass1_pipeline
            .as_ref()
            .ok_or("argmax_pass1 pipeline missing")?;
        let p2 = self
            .argmax_pass2_pipeline
            .as_ref()
            .ok_or("argmax_pass2 pipeline missing")?;
        let partials = self
            .argmax_partials_buf
            .as_ref()
            .ok_or("argmax partials buf missing")?;
        let result = self
            .argmax_result_buf
            .as_ref()
            .ok_or("argmax result buf missing")?;

        let n_parts = ((vocab_size + 255) / 256) as u32;
        let n = vocab_size as u32;
        {
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(p1);
            encoder.set_buffer(0, Some(logits_buf), 0);
            encoder.set_buffer(1, Some(partials), 0);
            encoder.set_bytes(2, 4, &n as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(
                MTLSize::new(n_parts as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
        }
        {
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(p2);
            encoder.set_buffer(0, Some(partials), 0);
            encoder.set_buffer(1, Some(result), 0);
            encoder.set_bytes(2, 4, &n_parts as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            encoder.end_encoding();
        }
        Ok(())
    }

    pub fn read_argmax_result(&self) -> u32 {
        unsafe { *(self.argmax_result_buf.as_ref().unwrap().contents() as *const u32) }
    }

    pub fn encode_swiglu(
        &self,
        cb: &CommandBufferRef,
        h1: &Buffer,
        h3: &Buffer,
        dim: usize,
    ) -> Result<(), String> {
        let pipeline = self.swiglu_pipeline.as_ref().ok_or("No SwiGLU pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(h1), 0);
        encoder.set_buffer(1, Some(h3), 0);
        let dim_u32 = dim as u32;
        encoder.set_bytes(2, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_rope(
        &self,
        cb: &CommandBufferRef,
        qb: &Buffer,
        kb: &Buffer,
        pos: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        scaling_factor: f32,
    ) -> Result<(), String> {
        let pipeline = self.rope_pipeline.as_ref().ok_or("No RoPE pipeline")?;
        let q_heads = (qb.length() as usize / 4) / head_dim;
        let k_heads = (kb.length() as usize / 4) / head_dim;
        let rotary_dim = rotary_dim.min(head_dim) & !1;
        if rotary_dim < 2 {
            return Ok(());
        }

        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_rotary_dim = rotary_dim as u32;
        let u_qh = q_heads as u32;
        let u_kh = k_heads as u32;
        encoder.set_bytes(2, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &u_rotary_dim as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &rope_theta as *const f32 as *const c_void);
        encoder.set_bytes(6, 4, &scaling_factor as *const f32 as *const c_void);
        encoder.set_bytes(7, 4, &u_qh as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &u_kh as *const u32 as *const c_void);
        let n_pairs = q_heads.max(k_heads) * (rotary_dim / 2);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_pairs as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    /// Encode RoPE using a precomputed `inv_freqs` buffer.
    /// `inv_freqs` must have length `rotary_dim / 2`.  Supports YaRN, LLaMA3-extended,
    /// and any other scaling mode whose per-dimension frequencies differ from base-theta.
    pub fn encode_rope_with_freqs(
        &self,
        cb: &CommandBufferRef,
        qb: &Buffer,
        kb: &Buffer,
        inv_freqs_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .rope_with_freqs_pipeline
            .as_ref()
            .ok_or("No rope_with_freqs pipeline")?;
        let q_heads = (qb.length() as usize / 4) / head_dim;
        let k_heads = (kb.length() as usize / 4) / head_dim;
        let rotary_dim = rotary_dim.min(head_dim) & !1;
        if rotary_dim < 2 {
            return Ok(());
        }
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_rdim = rotary_dim as u32;
        let u_qh = q_heads as u32;
        let u_kh = k_heads as u32;
        encoder.set_bytes(2, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &u_rdim as *const u32 as *const c_void);
        encoder.set_buffer(5, Some(inv_freqs_buf), 0);
        encoder.set_bytes(6, 4, &u_qh as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &u_kh as *const u32 as *const c_void);
        let n_pairs = q_heads.max(k_heads) * (rotary_dim / 2);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_pairs as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_split_gated_query(
        &self,
        cb: &CommandBufferRef,
        input: &Buffer,
        q_out: &Buffer,
        gate_out: &Buffer,
        dim: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .split_gated_query_pipeline
            .as_ref()
            .ok_or("No gated-query split pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(q_out), 0);
        encoder.set_buffer(2, Some(gate_out), 0);
        let dim_u32 = dim as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(3, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_mul_inplace(
        &self,
        cb: &CommandBufferRef,
        x: &Buffer,
        gate: &Buffer,
        dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .mul_inplace_pipeline
            .as_ref()
            .ok_or("No multiply pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(gate), 0);
        let dim_u32 = dim as u32;
        encoder.set_bytes(2, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    /// Encode a flash-attention-decode operation into `cb` (no commit/wait).
    /// KV cache buffers must hold F16 data (2 bytes per element).
    pub fn encode_attention(
        &self,
        cb: &CommandBufferRef,
        qb: &Buffer,
        kb: &Buffer,
        vb: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        out_buf: &Buffer,
        pos: usize,
        _dim: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        window_size: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .flash_attention_pipeline
            .as_ref()
            .ok_or("No Flash Attention pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        encoder.set_buffer(2, Some(vb), 0);
        encoder.set_buffer(3, Some(k_cache), 0);
        encoder.set_buffer(4, Some(v_cache), 0);
        encoder.set_buffer(5, Some(out_buf), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_nh = n_heads as u32;
        let u_nkvh = n_kv_heads as u32;
        let u_win = window_size as u32;
        encoder.set_bytes(6, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &u_nh as *const u32 as *const c_void);
        encoder.set_bytes(9, 4, &u_nkvh as *const u32 as *const c_void);
        encoder.set_bytes(10, 4, &u_win as *const u32 as *const c_void);
        // 128 threads per group (FA_THREADS in shader); one group per Q head
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        encoder.set_bytes(10, 4, &u_win as *const u32 as *const c_void);
        // 128 threads per group (FA_THREADS in shader); one group per Q head
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    /// Encode flash attention with Q8A KV cache (int8 + per-head-position f32 scale).
    /// k_cache / v_cache: [cache_len * kv_dim] i8
    /// k_scales / v_scales: [cache_len * n_kv_heads] f32
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_q8a(
        &self,
        cb: &CommandBufferRef,
        qb: &Buffer,
        kb: &Buffer,
        vb: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        k_scales: &Buffer,
        v_scales: &Buffer,
        out_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        window_size: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .flash_attention_q8a_pipeline
            .as_ref()
            .ok_or("No Q8A Flash Attention pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        encoder.set_buffer(2, Some(vb), 0);
        encoder.set_buffer(3, Some(k_cache), 0);
        encoder.set_buffer(4, Some(v_cache), 0);
        encoder.set_buffer(5, Some(k_scales), 0);
        encoder.set_buffer(6, Some(v_scales), 0);
        encoder.set_buffer(7, Some(out_buf), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_nh = n_heads as u32;
        let u_nkvh = n_kv_heads as u32;
        let u_win = window_size as u32;
        encoder.set_bytes(8, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(9, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(10, 4, &u_nh as *const u32 as *const c_void);
        encoder.set_bytes(11, 4, &u_nkvh as *const u32 as *const c_void);
        encoder.set_bytes(12, 4, &u_win as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_attention_q8a(
        &self,
        qb: &Buffer,
        kb: &Buffer,
        vb: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        k_scales: &Buffer,
        v_scales: &Buffer,
        out_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        window_size: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .flash_attention_q8a_pipeline
            .as_ref()
            .ok_or("No Q8A Flash Attention pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        encoder.set_buffer(2, Some(vb), 0);
        encoder.set_buffer(3, Some(k_cache), 0);
        encoder.set_buffer(4, Some(v_cache), 0);
        encoder.set_buffer(5, Some(k_scales), 0);
        encoder.set_buffer(6, Some(v_scales), 0);
        encoder.set_buffer(7, Some(out_buf), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_nh = n_heads as u32;
        let u_nkvh = n_kv_heads as u32;
        let u_win = window_size as u32;
        encoder.set_bytes(8, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(9, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(10, 4, &u_nh as *const u32 as *const c_void);
        encoder.set_bytes(11, 4, &u_nkvh as *const u32 as *const c_void);
        encoder.set_bytes(12, 4, &u_win as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    // ── Paged Q8A attention ───────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_paged_q8a(
        &self,
        cb: &CommandBufferRef,
        qb: &Buffer,
        kb: &Buffer,
        vb: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        k_scales: &Buffer,
        v_scales: &Buffer,
        block_table: &Buffer,
        out_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        p_start: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .flash_attention_paged_q8a_pipeline
            .as_ref()
            .ok_or("No paged Q8A pipeline")?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        encoder.set_buffer(2, Some(vb), 0);
        encoder.set_buffer(3, Some(k_cache), 0);
        encoder.set_buffer(4, Some(v_cache), 0);
        encoder.set_buffer(5, Some(k_scales), 0);
        encoder.set_buffer(6, Some(v_scales), 0);
        encoder.set_buffer(7, Some(block_table), 0);
        encoder.set_buffer(8, Some(out_buf), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_nh = n_heads as u32;
        let u_nkvh = n_kv_heads as u32;
        let u_pstart = p_start as u32;
        encoder.set_bytes(9, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(10, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(11, 4, &u_nh as *const u32 as *const c_void);
        encoder.set_bytes(12, 4, &u_nkvh as *const u32 as *const c_void);
        encoder.set_bytes(13, 4, &u_pstart as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_attention_paged_q8a(
        &self,
        qb: &Buffer,
        kb: &Buffer,
        vb: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        k_scales: &Buffer,
        v_scales: &Buffer,
        block_table: &Buffer,
        out_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        p_start: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .flash_attention_paged_q8a_pipeline
            .as_ref()
            .ok_or("No paged Q8A pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        encoder.set_buffer(2, Some(vb), 0);
        encoder.set_buffer(3, Some(k_cache), 0);
        encoder.set_buffer(4, Some(v_cache), 0);
        encoder.set_buffer(5, Some(k_scales), 0);
        encoder.set_buffer(6, Some(v_scales), 0);
        encoder.set_buffer(7, Some(block_table), 0);
        encoder.set_buffer(8, Some(out_buf), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_nh = n_heads as u32;
        let u_nkvh = n_kv_heads as u32;
        let u_pstart = p_start as u32;
        encoder.set_bytes(9, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(10, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(11, 4, &u_nh as *const u32 as *const c_void);
        encoder.set_bytes(12, 4, &u_nkvh as *const u32 as *const c_void);
        encoder.set_bytes(13, 4, &u_pstart as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn encode_linear_attn_block(
        &self,
        cb: &CommandBufferRef,
        input: &Buffer,
        conv_state: &Buffer,
        conv_weights: &Buffer,
        conv_bias: &Buffer,
        conv_output: &Buffer,
        z_in: &Buffer,
        beta_logits: &Buffer,
        a_in: &Buffer,
        a_log: &Buffer,
        dt_bias: &Buffer,
        norm_w: &Buffer,
        recurrent_state: &Buffer,
        output: &Buffer,
        kernel_size: usize,
        conv_dim: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let conv_pipeline = self
            .linear_attn_conv_pipeline
            .as_ref()
            .ok_or("No linear attention conv pipeline")?;
        let recurrent_pipeline = self
            .linear_attn_recurrent_pipeline
            .as_ref()
            .ok_or("No linear attention recurrent pipeline")?;

        let conv_encoder = cb.new_compute_command_encoder();
        conv_encoder.set_compute_pipeline_state(conv_pipeline);
        conv_encoder.set_buffer(0, Some(input), 0);
        conv_encoder.set_buffer(1, Some(conv_state), 0);
        conv_encoder.set_buffer(2, Some(conv_weights), 0);
        conv_encoder.set_buffer(3, Some(conv_bias), 0);
        conv_encoder.set_buffer(4, Some(conv_output), 0);
        let kernel_size_u32 = kernel_size as u32;
        let conv_dim_u32 = conv_dim as u32;
        conv_encoder.set_bytes(5, 4, &kernel_size_u32 as *const u32 as *const c_void);
        conv_encoder.set_bytes(6, 4, &conv_dim_u32 as *const u32 as *const c_void);
        conv_encoder.dispatch_thread_groups(
            MTLSize::new((conv_dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        conv_encoder.end_encoding();

        let recurrent_encoder = cb.new_compute_command_encoder();
        recurrent_encoder.set_compute_pipeline_state(recurrent_pipeline);
        recurrent_encoder.set_buffer(0, Some(conv_output), 0);
        recurrent_encoder.set_buffer(1, Some(z_in), 0);
        recurrent_encoder.set_buffer(2, Some(beta_logits), 0);
        recurrent_encoder.set_buffer(3, Some(a_in), 0);
        recurrent_encoder.set_buffer(4, Some(a_log), 0);
        recurrent_encoder.set_buffer(5, Some(dt_bias), 0);
        recurrent_encoder.set_buffer(6, Some(norm_w), 0);
        recurrent_encoder.set_buffer(7, Some(recurrent_state), 0);
        recurrent_encoder.set_buffer(8, Some(output), 0);
        let num_key_heads_u32 = num_key_heads as u32;
        let num_value_heads_u32 = num_value_heads as u32;
        let key_head_dim_u32 = key_head_dim as u32;
        let value_head_dim_u32 = value_head_dim as u32;
        recurrent_encoder.set_bytes(9, 4, &num_key_heads_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(10, 4, &num_value_heads_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(11, 4, &key_head_dim_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(12, 4, &value_head_dim_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(13, 4, &eps as *const f32 as *const c_void);
        recurrent_encoder.dispatch_thread_groups(
            MTLSize::new(num_value_heads as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
        recurrent_encoder.end_encoding();
        Ok(())
    }

    pub fn dispatch_rmsnorm_buffer(
        &self,
        x: &Buffer,
        w: &Buffer,
        out: &Buffer,
        dim: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .rmsnorm_pipeline
            .as_ref()
            .ok_or("No RMSNorm pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(out), 0);
        let dim_u32 = dim as u32;
        let zero_centered_u32 = if zero_centered { 1u32 } else { 0u32 };
        encoder.set_bytes(3, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        encoder.set_bytes(5, 4, &zero_centered_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_rmsnorm(&self, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, String> {
        let dim = x.len();
        if w.len() != dim {
            return Err("RMSNorm input and weight lengths differ".to_string());
        }
        let x_buf = self.prepare_f32_dense_tensor(x);
        let w_buf = self.prepare_f32_dense_tensor(w);
        let out_buf = self.device.new_buffer(
            (dim * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_rmsnorm_buffer(&x_buf, &w_buf, &out_buf, dim, eps, false)?;
        let mut out = vec![0.0f32; dim];
        unsafe {
            std::ptr::copy_nonoverlapping(out_buf.contents() as *const f32, out.as_mut_ptr(), dim);
        }
        Ok(out)
    }

    pub fn dispatch_rmsnorm_rows_buffer(
        &self,
        x: &Buffer,
        w: &Buffer,
        row_dim: usize,
        rows: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .rmsnorm_rows_pipeline
            .as_ref()
            .ok_or("No row RMSNorm pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        let row_dim_u32 = row_dim as u32;
        let rows_u32 = rows as u32;
        let zero_centered_u32 = if zero_centered { 1u32 } else { 0u32 };
        encoder.set_bytes(2, 4, &row_dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &rows_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        encoder.set_bytes(5, 4, &zero_centered_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn prepare_tensor(
        &self,
        data: &[u8],
        num_rows: usize,
        num_cols: usize,
        qt: QuantType,
    ) -> Result<MetalBuffer, String> {
        let blocks_per_row = (num_cols + 255) / 256;
        let total_blocks = num_rows * blocks_per_row;
        let gpu_stride = 144;

        if qt == QuantType::Q4HQ {
            let header = lns_core::parse_hq_payload_header(data, QuantType::Q4HQ)
                .map_err(|e| format!("invalid Q4HQ payload: {e}"))?;
            let expected_bytes = total_blocks * gpu_stride;
            let available_bytes = header.block_data_bytes as usize;
            if available_bytes < expected_bytes {
                return Err(format!(
                    "Q4HQ payload block data too short: expected at least {expected_bytes}, got {available_bytes}"
                ));
            }
            let start = header.block_data_off as usize;
            let end = start + expected_bytes;
            let block_data = &data[start..end];
            let buf = self.device.new_buffer_with_data(
                block_data.as_ptr() as *const _,
                block_data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            return Ok(MetalBuffer(buf, qt));
        }

        if qt != QuantType::Q4L {
            return Err(format!(
                "prepare_tensor: Only Q4L/Q4HQ supported, got {:?}",
                qt
            ));
        }
        let lns_stride = std::mem::size_of::<lns_core::Q4LSuperBlock>();

        let mut packed = vec![0u8; total_blocks * gpu_stride];
        for i in 0..total_blocks {
            let src = i * lns_stride;
            let dst = i * gpu_stride;
            if src + lns_stride > data.len() {
                break;
            }
            let g_bits = u16::from_le_bytes([data[src], data[src + 1]]);
            let g_f32 = F16::from_bits(g_bits).to_f32();
            for s_idx in 0..8usize {
                let byte_idx = s_idx / 2;
                let shift = (s_idx % 2) * 4;
                let sl = ((data[src + 2 + byte_idx] >> shift) & 0xF) as i32;
                let ef_f32 = g_f32 * (2.0f32).powi(sl - 7);
                let ef_bits = F16::from_f32(ef_f32).to_bits();
                let off = dst + s_idx * 2;
                packed[off..off + 2].copy_from_slice(&ef_bits.to_le_bytes());
            }
            packed[dst + 16..dst + 144].copy_from_slice(&data[src + 6..src + lns_stride]);
        }
        let buf = self.device.new_buffer_with_data(
            packed.as_ptr() as *const _,
            packed.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBuffer(buf, qt))
    }

    pub fn dispatch_gemv_cached(
        &self,
        weights: &MetalBuffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
        _qt: QuantType,
    ) -> Result<(), String> {
        let pipeline = self
            .gemv_pipelines
            .get(&weights.1)
            .ok_or("Pipeline missing")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weights.0), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(y), 0);
        let ncols = in_dim as u32;
        encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_gemv_batch_cached(
        &self,
        x: &Buffer,
        jobs: &[GemvBatchJob<'_>],
    ) -> Result<(), String> {
        if jobs.is_empty() {
            return Ok(());
        }

        let cb = self.queue.new_command_buffer();
        for job in jobs {
            let pipeline = self
                .gemv_pipelines
                .get(&job.weights.1)
                .ok_or("Pipeline missing")?;
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&job.weights.0), 0);
            encoder.set_buffer(1, Some(x), 0);
            encoder.set_buffer(2, Some(job.y), 0);
            let ncols = job.in_dim as u32;
            encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(
                MTLSize::new(job.out_dim as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_gemv_accumulate_cached(
        &self,
        weights: &MetalBuffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .gemv_accumulate_pipeline_for(weights.1)
            .ok_or("Accumulate pipeline missing")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weights.0), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(y), 0);
        let ncols = in_dim as u32;
        encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn encode_f16_gemv(
        &self,
        cb: &CommandBufferRef,
        weights: &Buffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        let encoder = cb.new_compute_command_encoder();
        if let Some(pipeline) = self.f16_gemv_multirow_pipeline.as_ref() {
            let ncols = in_dim as u32;
            let nrows = out_dim as u32;
            let n_tg = ((out_dim + 3) / 4) as u64;
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(weights), 0);
            encoder.set_buffer(1, Some(x), 0);
            encoder.set_buffer(2, Some(y), 0);
            encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &nrows as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
            encoder.end_encoding();
            return Ok(());
        }
        let pipeline = self
            .f16_gemv_pipeline
            .as_ref()
            .ok_or("No F16 GEMV pipeline")?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(y), 0);
        let ncols = in_dim as u32;
        encoder.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    pub fn dispatch_f16_gemv(
        &self,
        weights: &Buffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        let cb = self.queue.new_command_buffer();
        self.encode_f16_gemv(&cb, weights, x, y, out_dim, in_dim)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_swiglu(&self, h1: &Buffer, h3: &Buffer, dim: usize) -> Result<(), String> {
        let pipeline = self.swiglu_pipeline.as_ref().ok_or("No SwiGLU pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(h1), 0);
        encoder.set_buffer(1, Some(h3), 0);
        let dim_u32 = dim as u32;
        encoder.set_bytes(2, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_rope(
        &self,
        qb: &Buffer,
        kb: &Buffer,
        pos: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        scaling_factor: f32,
    ) -> Result<(), String> {
        let pipeline = self.rope_pipeline.as_ref().ok_or("No RoPE pipeline")?;
        let q_heads = (qb.length() as usize / 4) / head_dim;
        let k_heads = (kb.length() as usize / 4) / head_dim;
        let rotary_dim = rotary_dim.min(head_dim) & !1;
        if rotary_dim < 2 {
            return Ok(());
        }
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_rotary_dim = rotary_dim as u32;
        let u_qh = q_heads as u32;
        let u_kh = k_heads as u32;
        encoder.set_bytes(2, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &u_rotary_dim as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &rope_theta as *const f32 as *const c_void);
        encoder.set_bytes(6, 4, &scaling_factor as *const f32 as *const c_void);
        encoder.set_bytes(7, 4, &u_qh as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &u_kh as *const u32 as *const c_void);
        let n_pairs = q_heads.max(k_heads) * (rotary_dim / 2);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_pairs as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_rope_with_freqs(
        &self,
        qb: &Buffer,
        kb: &Buffer,
        inv_freqs_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(), String> {
        let cb = self.queue.new_command_buffer();
        self.encode_rope_with_freqs(&cb, qb, kb, inv_freqs_buf, pos, head_dim, rotary_dim)?;
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_split_gated_query(
        &self,
        input: &Buffer,
        q_out: &Buffer,
        gate_out: &Buffer,
        dim: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .split_gated_query_pipeline
            .as_ref()
            .ok_or("No gated-query split pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(q_out), 0);
        encoder.set_buffer(2, Some(gate_out), 0);
        let dim_u32 = dim as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(3, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_mul_inplace(
        &self,
        x: &Buffer,
        gate: &Buffer,
        dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .mul_inplace_pipeline
            .as_ref()
            .ok_or("No multiply pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(gate), 0);
        let dim_u32 = dim as u32;
        encoder.set_bytes(2, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_add_inplace(&self, x: &Buffer, y: &Buffer, dim: usize) -> Result<(), String> {
        let pipeline = self
            .add_inplace_pipeline
            .as_ref()
            .ok_or("No add pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(y), 0);
        let dim_u32 = dim as u32;
        encoder.set_bytes(2, 4, &dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    /// Dispatch a flash-attention-decode operation (commit + wait).
    /// KV cache buffers must hold F16 data (2 bytes per element).
    pub fn dispatch_attention(
        &self,
        qb: &Buffer,
        kb: &Buffer,
        vb: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        out_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        window_size: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .flash_attention_pipeline
            .as_ref()
            .ok_or("No Flash Attention pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qb), 0);
        encoder.set_buffer(1, Some(kb), 0);
        encoder.set_buffer(2, Some(vb), 0);
        encoder.set_buffer(3, Some(k_cache), 0);
        encoder.set_buffer(4, Some(v_cache), 0);
        encoder.set_buffer(5, Some(out_buf), 0);
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_nh = n_heads as u32;
        let u_nkvh = n_kv_heads as u32;
        let u_win = window_size as u32;
        encoder.set_bytes(6, 4, &u_pos as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &u_hdim as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &u_nh as *const u32 as *const c_void);
        encoder.set_bytes(9, 4, &u_nkvh as *const u32 as *const c_void);
        encoder.set_bytes(10, 4, &u_win as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_linear_attn_conv(
        &self,
        input: &Buffer,
        state: &Buffer,
        weights: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        kernel_size: usize,
        conv_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .linear_attn_conv_pipeline
            .as_ref()
            .ok_or("No linear attention conv pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(state), 0);
        encoder.set_buffer(2, Some(weights), 0);
        encoder.set_buffer(3, Some(bias), 0);
        encoder.set_buffer(4, Some(output), 0);
        let kernel_size_u32 = kernel_size as u32;
        let conv_dim_u32 = conv_dim as u32;
        encoder.set_bytes(5, 4, &kernel_size_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &conv_dim_u32 as *const u32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new((conv_dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_linear_attn_recurrent(
        &self,
        conv_out: &Buffer,
        z_in: &Buffer,
        beta_logits: &Buffer,
        a_in: &Buffer,
        a_log: &Buffer,
        dt_bias: &Buffer,
        norm_w: &Buffer,
        recurrent_state: &Buffer,
        output: &Buffer,
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let pipeline = self
            .linear_attn_recurrent_pipeline
            .as_ref()
            .ok_or("No linear attention recurrent pipeline")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(conv_out), 0);
        encoder.set_buffer(1, Some(z_in), 0);
        encoder.set_buffer(2, Some(beta_logits), 0);
        encoder.set_buffer(3, Some(a_in), 0);
        encoder.set_buffer(4, Some(a_log), 0);
        encoder.set_buffer(5, Some(dt_bias), 0);
        encoder.set_buffer(6, Some(norm_w), 0);
        encoder.set_buffer(7, Some(recurrent_state), 0);
        encoder.set_buffer(8, Some(output), 0);
        let num_key_heads_u32 = num_key_heads as u32;
        let num_value_heads_u32 = num_value_heads as u32;
        let key_head_dim_u32 = key_head_dim as u32;
        let value_head_dim_u32 = value_head_dim as u32;
        encoder.set_bytes(9, 4, &num_key_heads_u32 as *const u32 as *const c_void);
        encoder.set_bytes(10, 4, &num_value_heads_u32 as *const u32 as *const c_void);
        encoder.set_bytes(11, 4, &key_head_dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(12, 4, &value_head_dim_u32 as *const u32 as *const c_void);
        encoder.set_bytes(13, 4, &eps as *const f32 as *const c_void);
        encoder.dispatch_thread_groups(
            MTLSize::new(num_value_heads as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    pub fn dispatch_linear_attn_block(
        &self,
        input: &Buffer,
        conv_state: &Buffer,
        conv_weights: &Buffer,
        conv_bias: &Buffer,
        conv_output: &Buffer,
        z_in: &Buffer,
        beta_logits: &Buffer,
        a_in: &Buffer,
        a_log: &Buffer,
        dt_bias: &Buffer,
        norm_w: &Buffer,
        recurrent_state: &Buffer,
        output: &Buffer,
        kernel_size: usize,
        conv_dim: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let conv_pipeline = self
            .linear_attn_conv_pipeline
            .as_ref()
            .ok_or("No linear attention conv pipeline")?;
        let recurrent_pipeline = self
            .linear_attn_recurrent_pipeline
            .as_ref()
            .ok_or("No linear attention recurrent pipeline")?;
        let cb = self.queue.new_command_buffer();

        let conv_encoder = cb.new_compute_command_encoder();
        conv_encoder.set_compute_pipeline_state(conv_pipeline);
        conv_encoder.set_buffer(0, Some(input), 0);
        conv_encoder.set_buffer(1, Some(conv_state), 0);
        conv_encoder.set_buffer(2, Some(conv_weights), 0);
        conv_encoder.set_buffer(3, Some(conv_bias), 0);
        conv_encoder.set_buffer(4, Some(conv_output), 0);
        let kernel_size_u32 = kernel_size as u32;
        let conv_dim_u32 = conv_dim as u32;
        conv_encoder.set_bytes(5, 4, &kernel_size_u32 as *const u32 as *const c_void);
        conv_encoder.set_bytes(6, 4, &conv_dim_u32 as *const u32 as *const c_void);
        conv_encoder.dispatch_thread_groups(
            MTLSize::new((conv_dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        conv_encoder.end_encoding();

        let recurrent_encoder = cb.new_compute_command_encoder();
        recurrent_encoder.set_compute_pipeline_state(recurrent_pipeline);
        recurrent_encoder.set_buffer(0, Some(conv_output), 0);
        recurrent_encoder.set_buffer(1, Some(z_in), 0);
        recurrent_encoder.set_buffer(2, Some(beta_logits), 0);
        recurrent_encoder.set_buffer(3, Some(a_in), 0);
        recurrent_encoder.set_buffer(4, Some(a_log), 0);
        recurrent_encoder.set_buffer(5, Some(dt_bias), 0);
        recurrent_encoder.set_buffer(6, Some(norm_w), 0);
        recurrent_encoder.set_buffer(7, Some(recurrent_state), 0);
        recurrent_encoder.set_buffer(8, Some(output), 0);
        let num_key_heads_u32 = num_key_heads as u32;
        let num_value_heads_u32 = num_value_heads as u32;
        let key_head_dim_u32 = key_head_dim as u32;
        let value_head_dim_u32 = value_head_dim as u32;
        recurrent_encoder.set_bytes(9, 4, &num_key_heads_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(10, 4, &num_value_heads_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(11, 4, &key_head_dim_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(12, 4, &value_head_dim_u32 as *const u32 as *const c_void);
        recurrent_encoder.set_bytes(13, 4, &eps as *const f32 as *const c_void);
        recurrent_encoder.dispatch_thread_groups(
            MTLSize::new(num_value_heads as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
        recurrent_encoder.end_encoding();

        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }
}

// ── Persistent-encoder dispatch methods ──────────────────────────────────────
// Each `*_enc` variant takes an already-open `&ComputeCommandEncoderRef`.
// The caller inserts `MetalContext::barrier` between dependent dispatches and
// calls `end_encoding()` exactly once.  This eliminates the per-kernel
// encoder alloc/end overhead in `forward_single_cb`.

impl MetalContext {
    /// Barrier on a slice of `&Buffer` inside an open compute encoder.
    #[inline]
    pub fn barrier(enc: &ComputeCommandEncoderRef, bufs: &[&Buffer]) {
        match bufs {
            [] => {}
            [a] => {
                let rs: [&ResourceRef; 1] = [&***a];
                enc.memory_barrier_with_resources(&rs);
            }
            [a, b] => {
                let rs: [&ResourceRef; 2] = [&***a, &***b];
                enc.memory_barrier_with_resources(&rs);
            }
            [a, b, c] => {
                let rs: [&ResourceRef; 3] = [&***a, &***b, &***c];
                enc.memory_barrier_with_resources(&rs);
            }
            [a, b, c, d] => {
                let rs: [&ResourceRef; 4] = [&***a, &***b, &***c, &***d];
                enc.memory_barrier_with_resources(&rs);
            }
            [a, b, c, d, e] => {
                let rs: [&ResourceRef; 5] = [&***a, &***b, &***c, &***d, &***e];
                enc.memory_barrier_with_resources(&rs);
            }
            [a, b, c, d, e, f] => {
                let rs: [&ResourceRef; 6] = [&***a, &***b, &***c, &***d, &***e, &***f];
                enc.memory_barrier_with_resources(&rs);
            }
            _ => {
                let rs: Vec<&ResourceRef> = bufs.iter().map(|b| -> &ResourceRef { &***b }).collect();
                enc.memory_barrier_with_resources(&rs);
            }
        }
    }

    pub fn rmsnorm_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        w: &Buffer,
        out: &Buffer,
        dim: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .rmsnorm_pipeline
            .as_ref()
            .ok_or("No RMSNorm pipeline")?;
        let dim_u32 = dim as u32;
        let zc_u32 = if zero_centered { 1u32 } else { 0u32 };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &dim_u32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &zc_u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    pub fn rmsnorm_rows_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        w: &Buffer,
        row_dim: usize,
        rows: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .rmsnorm_rows_pipeline
            .as_ref()
            .ok_or("No row RMSNorm pipeline")?;
        let rd_u32 = row_dim as u32;
        let rows_u32 = rows as u32;
        let zc_u32 = if zero_centered { 1u32 } else { 0u32 };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_bytes(2, 4, &rd_u32 as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &rows_u32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &zc_u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    /// Dispatch all batch jobs to an already-open encoder (no per-job encoder overhead).
    pub fn gemv_batch_multirow_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        jobs: &[GemvBatchJob<'_>],
    ) -> Result<(), String> {
        for job in jobs {
            let quant_type = job.weights.1;
            let Some(pipeline) = self.gemv_multirow_pipeline_for(quant_type) else {
                let fb = self
                    .gemv_pipelines
                    .get(&quant_type)
                    .ok_or("Pipeline missing")?;
                enc.set_compute_pipeline_state(fb);
                enc.set_buffer(0, Some(&job.weights.0), 0);
                enc.set_buffer(1, Some(x), 0);
                enc.set_buffer(2, Some(job.y), 0);
                let ncols = job.in_dim as u32;
                enc.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(job.out_dim as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                continue;
            };
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&job.weights.0), 0);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(job.y), 0);
            let ncols = job.in_dim as u32;
            let nrows = job.out_dim as u32;
            let n_tg = ((job.out_dim + 3) / 4) as u64;
            enc.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nrows as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        }
        Ok(())
    }

    pub fn gemv_accumulate_multirow_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        weights: &MetalBuffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        let ncols = in_dim as u32;
        let Some(pipeline) = self.gemv_accumulate_multirow_pipeline_for(weights.1) else {
            let fallback = self
                .gemv_accumulate_pipeline_for(weights.1)
                .ok_or("Accumulate pipeline missing")?;
            enc.set_compute_pipeline_state(fallback);
            enc.set_buffer(0, Some(&weights.0), 0);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
            return Ok(());
        };
        let nrows = out_dim as u32;
        let n_tg = ((out_dim + 3) / 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&weights.0), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nrows as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    pub fn split_gated_query_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &Buffer,
        q_out: &Buffer,
        gate_out: &Buffer,
        dim: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .split_gated_query_pipeline
            .as_ref()
            .ok_or("No gated-query split pipeline")?;
        let dim_u32 = dim as u32;
        let head_dim_u32 = head_dim as u32;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(q_out), 0);
        enc.set_buffer(2, Some(gate_out), 0);
        enc.set_bytes(3, 4, &dim_u32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }

    pub fn rope_with_freqs_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        qb: &Buffer,
        kb: &Buffer,
        inv_freqs_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .rope_with_freqs_pipeline
            .as_ref()
            .ok_or("No rope_with_freqs pipeline")?;
        let q_heads = (qb.length() as usize / 4) / head_dim;
        let k_heads = (kb.length() as usize / 4) / head_dim;
        let rotary_dim = rotary_dim.min(head_dim) & !1;
        if rotary_dim < 2 {
            return Ok(());
        }
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_rdim = rotary_dim as u32;
        let u_qh = q_heads as u32;
        let u_kh = k_heads as u32;
        let n_pairs = q_heads.max(k_heads) * (rotary_dim / 2);
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(qb), 0);
        enc.set_buffer(1, Some(kb), 0);
        enc.set_bytes(2, 4, &u_pos as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &u_hdim as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &u_rdim as *const u32 as *const c_void);
        enc.set_buffer(5, Some(inv_freqs_buf), 0);
        enc.set_bytes(6, 4, &u_qh as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &u_kh as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((n_pairs as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_paged_q8a_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        qb: &Buffer,
        kb: &Buffer,
        vb: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        k_scales: &Buffer,
        v_scales: &Buffer,
        block_table: &Buffer,
        out_buf: &Buffer,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        p_start: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .flash_attention_paged_q8a_pipeline
            .as_ref()
            .ok_or("No paged Q8A pipeline")?;
        let u_pos = pos as u32;
        let u_hdim = head_dim as u32;
        let u_nh = n_heads as u32;
        let u_nkvh = n_kv_heads as u32;
        let u_pstart = p_start as u32;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(qb), 0);
        enc.set_buffer(1, Some(kb), 0);
        enc.set_buffer(2, Some(vb), 0);
        enc.set_buffer(3, Some(k_cache), 0);
        enc.set_buffer(4, Some(v_cache), 0);
        enc.set_buffer(5, Some(k_scales), 0);
        enc.set_buffer(6, Some(v_scales), 0);
        enc.set_buffer(7, Some(block_table), 0);
        enc.set_buffer(8, Some(out_buf), 0);
        enc.set_bytes(9, 4, &u_pos as *const u32 as *const c_void);
        enc.set_bytes(10, 4, &u_hdim as *const u32 as *const c_void);
        enc.set_bytes(11, 4, &u_nh as *const u32 as *const c_void);
        enc.set_bytes(12, 4, &u_nkvh as *const u32 as *const c_void);
        enc.set_bytes(13, 4, &u_pstart as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    pub fn mul_inplace_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        gate: &Buffer,
        dim: usize,
    ) -> Result<(), String> {
        let pipeline = self
            .mul_inplace_pipeline
            .as_ref()
            .ok_or("No multiply pipeline")?;
        let dim_u32 = dim as u32;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gate), 0);
        enc.set_bytes(2, 4, &dim_u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }

    pub fn swiglu_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        h1: &Buffer,
        h3: &Buffer,
        dim: usize,
    ) -> Result<(), String> {
        let pipeline = self.swiglu_pipeline.as_ref().ok_or("No SwiGLU pipeline")?;
        let dim_u32 = dim as u32;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(h1), 0);
        enc.set_buffer(1, Some(h3), 0);
        enc.set_bytes(2, 4, &dim_u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Ok(())
    }

    pub fn rmsnorm_gemv_multirow_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        norm_w: &Buffer,
        weights: &MetalBuffer,
        y: &Buffer,
        dim: usize,
        out_rows: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        if weights.1 != QuantType::Q4L {
            return Err("RMSNorm+GEMV multirow supports Q4L only".to_string());
        }
        let pipeline = self
            .q4l_rmsnorm_gemv_multirow_pipeline
            .as_ref()
            .ok_or("RMSNorm+GEMV multirow pipeline missing")?;
        let u_dim = dim as u32;
        let u_out_rows = out_rows as u32;
        let u_zc = if zero_centered { 1u32 } else { 0u32 };
        let n_tg = ((out_rows + 3) / 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(norm_w), 0);
        enc.set_buffer(2, Some(&weights.0), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, 4, &u_dim as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &u_out_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(7, 4, &u_zc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    /// Fused RMSNorm + Q4HQ GEMV for the lm_head projection.
    /// Same dispatch signature as `rmsnorm_gemv_multirow_enc` but uses the Q4HQ decode path.
    pub fn q4hq_rmsnorm_gemv_multirow_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        norm_w: &Buffer,
        weights: &MetalBuffer,
        y: &Buffer,
        dim: usize,
        out_rows: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<(), String> {
        let pipeline = self
            .q4hq_rmsnorm_gemv_multirow_pipeline
            .as_ref()
            .ok_or("q4hq_rmsnorm_gemv_multirow pipeline missing")?;
        let u_dim = dim as u32;
        let u_out_rows = out_rows as u32;
        let u_zc = if zero_centered { 1u32 } else { 0u32 };
        let n_tg = ((out_rows + 3) / 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(norm_w), 0);
        enc.set_buffer(2, Some(&weights.0), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, 4, &u_dim as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &u_out_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(7, 4, &u_zc as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    /// Fused W1+W3+SwiGLU: computes h_out[i] = silu(w1·x)[i] * (w3·x)[i] in one dispatch.
    /// Replaces separate w1_gemv + w3_gemv + swiglu dispatches (saves 2 kernel launches).
    /// `quant_type` selects Q4L vs Q4HQ decode path.
    pub fn w1w3_swiglu_multirow_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        w1: &MetalBuffer,
        w3: &MetalBuffer,
        x: &Buffer,
        h_out: &Buffer,
        hidden_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        if w1.1 != w3.1 {
            return Err(format!(
                "w1w3_swiglu_multirow: quant mismatch w1={:?} w3={:?}",
                w1.1, w3.1
            ));
        }
        let pipeline = match w1.1 {
            QuantType::Q4HQ => self
                .q4hq_w1w3_swiglu_multirow_pipeline
                .as_ref()
                .ok_or("q4hq_w1w3_swiglu_multirow pipeline missing")?,
            QuantType::Q4L => self
                .q4l_w1w3_swiglu_multirow_pipeline
                .as_ref()
                .ok_or("q4l_w1w3_swiglu_multirow pipeline missing")?,
            qt => return Err(format!("w1w3_swiglu_multirow: unsupported quant {:?}", qt)),
        };
        let ncols = in_dim as u32;
        let nrows = hidden_dim as u32;
        let n_tg = ((hidden_dim + 3) / 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&w1.0), 0);
        enc.set_buffer(1, Some(&w3.0), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(h_out), 0);
        enc.set_bytes(4, 4, &ncols as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &nrows as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    pub fn f16_gemv_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        weights: &Buffer,
        x: &Buffer,
        y: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), String> {
        // Prefer the multi-row variant (4 rows/TG with shared `x`) when
        // available — 4× fewer threadgroups and ~95 % less x-bandwidth than
        // the legacy 1-row/TG kernel.
        if let Some(pipeline) = self.f16_gemv_multirow_pipeline.as_ref() {
            let ncols = in_dim as u32;
            let nrows = out_dim as u32;
            let n_tg = ((out_dim + 3) / 4) as u64;
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(weights), 0);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nrows as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
            return Ok(());
        }
        let pipeline = self
            .f16_gemv_pipeline
            .as_ref()
            .ok_or("No F16 GEMV pipeline")?;
        let ncols = in_dim as u32;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(weights), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &ncols as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    /// Argmax two-pass (pass1 + internal barrier + pass2) in an open encoder.
    pub fn argmax_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        logits_buf: &Buffer,
        vocab_size: usize,
    ) -> Result<(), String> {
        let p1 = self
            .argmax_pass1_pipeline
            .as_ref()
            .ok_or("argmax_pass1 pipeline missing")?;
        let p2 = self
            .argmax_pass2_pipeline
            .as_ref()
            .ok_or("argmax_pass2 pipeline missing")?;
        let partials = self
            .argmax_partials_buf
            .as_ref()
            .ok_or("argmax partials buf missing")?;
        let result = self
            .argmax_result_buf
            .as_ref()
            .ok_or("argmax result buf missing")?;
        let n_parts = ((vocab_size + 255) / 256) as u32;
        let n = vocab_size as u32;
        enc.set_compute_pipeline_state(p1);
        enc.set_buffer(0, Some(logits_buf), 0);
        enc.set_buffer(1, Some(partials), 0);
        enc.set_bytes(2, 4, &n as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_parts as u64, 1, 1), MTLSize::new(256, 1, 1));
        Self::barrier(enc, &[partials]);
        enc.set_compute_pipeline_state(p2);
        enc.set_buffer(0, Some(partials), 0);
        enc.set_buffer(1, Some(result), 0);
        enc.set_bytes(2, 4, &n_parts as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        Ok(())
    }

    /// Linear-attention conv + recurrent (with internal barrier) in an open encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_attn_block_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &Buffer,
        conv_state: &Buffer,
        conv_weights: &Buffer,
        conv_bias: &Buffer,
        conv_output: &Buffer,
        z_in: &Buffer,
        beta_logits: &Buffer,
        a_in: &Buffer,
        a_log: &Buffer,
        dt_bias: &Buffer,
        norm_w: &Buffer,
        recurrent_state: &Buffer,
        output: &Buffer,
        kernel_size: usize,
        conv_dim: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let conv_pipeline = self
            .linear_attn_conv_pipeline
            .as_ref()
            .ok_or("No linear attention conv pipeline")?;
        let recurrent_pipeline = self
            .linear_attn_recurrent_pipeline
            .as_ref()
            .ok_or("No linear attention recurrent pipeline")?;
        let ksz_u32 = kernel_size as u32;
        let cdim_u32 = conv_dim as u32;
        enc.set_compute_pipeline_state(conv_pipeline);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(conv_state), 0);
        enc.set_buffer(2, Some(conv_weights), 0);
        enc.set_buffer(3, Some(conv_bias), 0);
        enc.set_buffer(4, Some(conv_output), 0);
        enc.set_bytes(5, 4, &ksz_u32 as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &cdim_u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((conv_dim as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        Self::barrier(enc, &[conv_output]);
        let nkh_u32 = num_key_heads as u32;
        let nvh_u32 = num_value_heads as u32;
        let khd_u32 = key_head_dim as u32;
        let vhd_u32 = value_head_dim as u32;
        enc.set_compute_pipeline_state(recurrent_pipeline);
        enc.set_buffer(0, Some(conv_output), 0);
        enc.set_buffer(1, Some(z_in), 0);
        enc.set_buffer(2, Some(beta_logits), 0);
        enc.set_buffer(3, Some(a_in), 0);
        enc.set_buffer(4, Some(a_log), 0);
        enc.set_buffer(5, Some(dt_bias), 0);
        enc.set_buffer(6, Some(norm_w), 0);
        enc.set_buffer(7, Some(recurrent_state), 0);
        enc.set_buffer(8, Some(output), 0);
        enc.set_bytes(9, 4, &nkh_u32 as *const u32 as *const c_void);
        enc.set_bytes(10, 4, &nvh_u32 as *const u32 as *const c_void);
        enc.set_bytes(11, 4, &khd_u32 as *const u32 as *const c_void);
        enc.set_bytes(12, 4, &vhd_u32 as *const u32 as *const c_void);
        enc.set_bytes(13, 4, &eps as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_value_heads as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
        Ok(())
    }
}

pub fn bench_q4l_decode(
    _blocks: &[lns_core::Q4LSuperBlock],
    n: usize,
    iters: usize,
    backend: ComputeBackend,
) -> Result<DecodeBenchResult, String> {
    let start = std::time::Instant::now();
    let mut checksum = 0.0f32;
    if backend == ComputeBackend::Cpu {
        for _ in 0..iters {
            let decoded = lns_core::dequantize_q4l(_blocks, n);
            checksum = decoded.iter().sum();
        }
    }
    Ok(DecodeBenchResult {
        elapsed_secs: start.elapsed().as_secs_f64(),
        weights_processed: n * iters,
        checksum,
    })
}

pub fn decode_quality_report(
    blocks: &[lns_core::Q4LSuperBlock],
    n_elements: usize,
) -> QualityReport {
    let decoded = lns_core::dequantize_q4l(blocks, n_elements);
    let mut zeros = 0usize;
    let mut hist = [0u64; 16];
    for b in blocks {
        for i in 0..256 {
            let (_, m) = b.get_weight(i);
            if m == 0 {
                zeros += 1;
            }
            let nibble = (b.weights[i / 2] >> ((i % 2) * 4)) & 0xF;
            hist[nibble as usize] += 1;
        }
    }
    let mut entropy = 0.0f64;
    let total_w = (blocks.len() * 256) as f64;
    for &count in &hist {
        if count > 0 {
            let p = count as f64 / total_w;
            entropy -= p * p.log2();
        }
    }
    let re_encoded = lns_core::quantize_q4l(&decoded);
    let roundtrip = lns_core::dequantize_q4l(&re_encoded, n_elements);
    let mut mse = 0.0f64;
    let mut var_signal = 0.0f64;
    for (orig, rt) in decoded.iter().zip(roundtrip.iter()) {
        let diff = (orig - rt) as f64;
        mse += diff * diff;
        var_signal += (*orig as f64) * (*orig as f64);
    }
    mse /= n_elements as f64;
    var_signal /= n_elements as f64;
    let snr = if mse > 0.0 {
        10.0 * (var_signal / mse).log10()
    } else {
        99.0
    };
    QualityReport {
        zero_fraction: zeros as f64 / (blocks.len() * 256) as f64,
        code_entropy_bits: entropy,
        roundtrip_rmse: mse.sqrt(),
        roundtrip_snr_db: snr,
        n_weights: n_elements,
    }
}

fn lns_decode_kernel_source() -> String {
    include_str!("shaders/lns_decode.metal").to_string()
}
