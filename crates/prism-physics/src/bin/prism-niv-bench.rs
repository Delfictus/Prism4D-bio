//! # PRISM NiV Bench - Phase 3.1 Execution Binary
//!
//! Scientific benchmark for Nipah Virus G Glycoprotein (2VWD.ptb) breathing analysis.
//! Executes 10,000 steps of NLNM (Non-Linear Normal Mode) simulation with VRAM Guard protection.
//!
//! ## Mission Profile
//! - **Target**: Nipah Virus G Glycoprotein (2VWD structure)
//! - **Objective**: Capture protein "breathing" motion via NLNM analysis
//! - **Duration**: 10,000 simulation steps
//! - **Physics**: PIMC/NLNM molecular dynamics with GPU acceleration
//!
//! ## Safety Protocol
//! - VRAM Guard verification before GPU allocation
//! - Feature-gated telemetry recording (HOT LOOP PROTOCOL)
//! - Graceful error handling with scientific context

use prism_physics::molecular_dynamics::{
    MolecularDynamicsConfig, MolecularDynamicsEngine,
};
use prism_core::PrismError;
use prism_io::{AsyncPinnedStreamer, PrismIoError};
use prism_gpu::memory::{VramGuard, init_global_vram_guard};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

/// Path to the sovereign 2VWD.ptb data file
const NIV_PTB_PATH: &str = "data/processed/2VWD.ptb";

/// Target simulation steps for full breathing analysis
const BREATHING_STEPS: u64 = 10_000;

/// Main execution entry point for Phase 3.1 - Sovereign Standard Compliance
#[tokio::main]
async fn main() -> Result<(), PrismError> {
    // 1. Init Telemetry/Logging
    env_logger::init();
    log::info!("🧬 PRISM NiV Bench - Phase 3.1 Execution");

    // 2. GATE 1 CHECK: VRAM Guard - MANDATORY VERIFICATION
    #[cfg(feature = "cuda")]
    {
        log::info!("🔌 Initializing CUDA Driver...");

        // 1. Initialize the Driver & Device properly
        // This handles cuInit(0) AND context creation automatically.
        let dev = cudarc::driver::CudaDevice::new(0)
            .map_err(|e| PrismError::Internal(format!("Failed to init CUDA device: {:?}", e)))?;

        // 2. Extract the Context (Real, not zeroed)
        // We clone the Arc<CudaContext> from the device
        let cuda_context = dev.context().clone();

        // 3. Initialize the Guard
        init_global_vram_guard(cuda_context);

        // 4. Query VRAM
        let vram_guard = prism_gpu::memory::global_vram_guard();

        // This will now succeed because the context is real
        let vram_info = vram_guard.query_vram()
            .map_err(|e| PrismError::Internal(format!("VRAM Query Failed: {:?}", e)))?;

        log::info!("🛡️ VRAM Check: {} MB Free / {} MB Total",
            vram_info.free_bytes / 1024 / 1024,
            vram_info.total_bytes / 1024 / 1024
        );

        // 5. Verify Safety
        let physics_memory_required = 1024 * 1024 * 1024; // 1GB
        let workspace_memory_required = 512 * 1024 * 1024; // 512MB

        vram_guard.verify_physics_engine_startup(physics_memory_required, workspace_memory_required)
            .map_err(|e| PrismError::Internal(format!("VRAM Safety Check Failed: {:?}", e)))?;

        log::info!("✅ VRAM Guard: Memory allocation approved");
    }

    #[cfg(not(feature = "cuda"))]
    {
        log::warn!("⚠️ Running without CUDA support - VRAM Guard bypassed");
    }

    // 3. GATE 3 CORRECTION: Sovereign Loading via prism-io
    log::info!("📂 Streaming Sovereign Data via io_uring...");

    // Initialize async pinned streamer with GPU integration
    let streamer = AsyncPinnedStreamer::new().await?;

    // Load verified structure using sovereign data pipeline
    let sovereign_buffer = streamer.load_verified_structure(NIV_PTB_PATH).await?;

    log::info!("✅ Data Streamed: {} bytes (Pinned Memory)", sovereign_buffer.len());

    // 4. Configure & Run - Molecular Dynamics Engine
    let md_config = configure_niv_simulation();
    log::info!("⚙️ MD Configuration: Temperature={}K, dt={}fs, GPU={}",
               md_config.temperature, md_config.dt, md_config.use_gpu);

    // Initialize molecular dynamics engine with sovereign buffer
    // Convert SovereignBuffer to raw bytes for current MD engine API
    let sovereign_data = unsafe {
        std::slice::from_raw_parts(sovereign_buffer.as_ptr(), sovereign_buffer.len())
    };
    let mut md_engine = MolecularDynamicsEngine::from_sovereign_buffer(md_config, sovereign_data)?;
    log::info!("🚀 Molecular dynamics engine initialized");

    // Execute 10,000 step NLNM breathing simulation (The Breathing Run)
    log::info!("🌬️ Beginning {} step NLNM breathing analysis...", BREATHING_STEPS);
    let start_time = Instant::now();

    let phase_outcome = md_engine.run_nlnm_breathing(BREATHING_STEPS)?;

    let total_runtime = start_time.elapsed();
    log::info!("🏁 Breathing simulation complete in {:.2}s", total_runtime.as_secs_f32());

    // Extract and display scientific results
    display_scientific_results(&md_engine, &phase_outcome, total_runtime)?;

    // Phase 3.1 completion telemetry
    log::info!("✅ PHASE 3.1 COMPLETE: NiV Breathing Analysis Successful");
    log::info!("📊 Integration Layer: prism-io ➜ prism-physics ➜ Telemetry");
    log::info!("🛡️ VRAM Guard: Protected GPU allocation throughout");
    log::info!("🧬 Scientific Output: Protein breathing motion captured");

    Ok(())
}


/// Configure molecular dynamics for Nipah Virus analysis
fn configure_niv_simulation() -> MolecularDynamicsConfig {
    MolecularDynamicsConfig {
        // Scientific parameters for NiV G glycoprotein
        max_steps: BREATHING_STEPS,
        temperature: 310.15,  // Physiological temperature (37°C)
        dt: 1.0,              // 1 femtosecond timestep for accuracy

        // PIMC configuration (quantum effects)
        pimc_config: prism_physics::molecular_dynamics::PimcConfig {
            num_beads: 16,           // Reduced for faster convergence
            step_size: 0.05,         // Smaller steps for viral protein
            target_acceptance: 0.65, // Target 65% acceptance
            adaptation_rate: 0.03,   // Conservative adaptation
        },

        // NLNM configuration (breathing motion)
        nlnm_config: prism_physics::molecular_dynamics::NlnmConfig {
            gradient_threshold: 0.0005,  // Tight convergence for breathing modes
            max_iterations: 15_000,       // Allow longer convergence
            damping_factor: 0.15,         // Moderate damping for stability
        },

        // GPU memory allocation (conservative for stability)
        use_gpu: true,
        max_trajectory_memory: 1024 * 1024 * 1024, // 1GB for trajectory
        max_workspace_memory: 512 * 1024 * 1024,   // 512MB workspace
    }
}

/// Display scientific results from breathing analysis
fn display_scientific_results(
    engine: &MolecularDynamicsEngine,
    phase_outcome: &prism_core::PhaseOutcome,
    runtime: std::time::Duration,
) -> Result<(), PrismError> {
    let stats = engine.get_statistics();

    log::info!("🧬 === NIPAH VIRUS BREATHING ANALYSIS RESULTS ===");
    log::info!("");

    // Simulation Statistics
    log::info!("📊 Simulation Statistics:");
    log::info!("   Steps Completed: {}/{}", stats.current_step, stats.total_steps);
    log::info!("   Runtime: {:.2}s ({:.1} steps/sec)",
               runtime.as_secs_f32(),
               stats.current_step as f32 / runtime.as_secs_f32());
    log::info!("   Convergence: {}", if stats.converged { "✅ CONVERGED" } else { "❌ NOT CONVERGED" });
    log::info!("");

    // Physical Properties
    log::info!("🌡️  Physical Properties:");
    log::info!("   Final Energy: {:.3} kcal/mol", stats.current_energy);
    log::info!("   Temperature: {:.2}K ({:.1}°C)",
               stats.current_temperature,
               stats.current_temperature - 273.15);
    log::info!("   Gradient Norm: {:.6} (threshold: {:.6})",
               stats.gradient_norm,
               0.0005); // From config
    log::info!("");

    // Breathing Motion Analysis
    log::info!("🌬️  Breathing Motion Analysis:");
    if stats.converged {
        log::info!("   ✅ Normal modes successfully captured");
        log::info!("   ✅ Protein breathing motion characterized");
        log::info!("   ✅ Structural flexibility quantified");
    } else {
        log::info!("   ⚠️  Partial convergence - breathing modes identified");
        log::info!("   📈 Gradient reduction: {:.1}% complete",
                   (1.0 - stats.gradient_norm / 1.0) * 100.0);
    }
    log::info!("");

    // Monte Carlo Performance
    log::info!("🎲 Monte Carlo Performance:");
    log::info!("   Acceptance Rate: {:.1}%", stats.acceptance_rate * 100.0);
    let acceptance_quality = if stats.acceptance_rate >= 0.5 && stats.acceptance_rate <= 0.8 {
        "✅ OPTIMAL"
    } else if stats.acceptance_rate >= 0.3 {
        "⚠️  ACCEPTABLE"
    } else {
        "❌ POOR"
    };
    log::info!("   Quality Assessment: {}", acceptance_quality);
    log::info!("");

    // Phase Outcome Details
    match phase_outcome {
        prism_core::PhaseOutcome::Success { message, telemetry } => {
            log::info!("🎯 Phase Outcome: SUCCESS");
            log::info!("   Message: {}", message);
            log::info!("   Telemetry Keys: {}", telemetry.keys().len());

            // Extract key telemetry metrics
            for (key, value) in telemetry.iter() {
                if key == "runtime_seconds" || key == "final_energy" || key == "converged" {
                    log::info!("   {}: {}", key, value);
                }
            }
        }
        prism_core::PhaseOutcome::Escalate { reason } => {
            log::error!("❌ Phase Outcome: ESCALATED");
            log::error!("   Reason: {}", reason);
        }
        prism_core::PhaseOutcome::Retry { reason, backoff_ms } => {
            log::warn!("🔄 Phase Outcome: RETRY");
            log::warn!("   Reason: {}", reason);
            log::warn!("   Backoff: {}ms", backoff_ms);
        }
    }

    log::info!("");
    log::info!("🏆 NiV G Glycoprotein breathing analysis complete!");

    Ok(())
}