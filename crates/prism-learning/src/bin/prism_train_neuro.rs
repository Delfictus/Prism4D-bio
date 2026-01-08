//! PRISM-Zero v3.1 Neuromorphic Training Binary
//!
//! Uses the Flashbulb Reservoir (E/I balanced SNN) instead of DQN.
//! NO PYTORCH REQUIRED - Pure Rust + CUDA.
//!
//! Architecture:
//! - 80/20 Excitatory/Inhibitory balance
//! - Adaptive time constants (5-50ms, matching protein dynamics)
//! - Reward-modulated RLS learning (Flashbulb plasticity)
//! - Velocity features for temporal dynamics (23→46 dim expansion)
//!
//! Recommended Config (vs DQN):
//! - DQN: 10 chunks × 100K steps = 1M total
//! - Dendritic: 100 chunks × 10K steps = 1M total (more learning signal)

use anyhow::{Context, Result};
use clap::Parser;
use log::{info, warn, debug};
use std::path::Path;
use std::time::Instant;
use std::fs;

use prism_learning::{
    CalibrationManifest,
    FeatureExtractor,
    DendriticAgent,
    DendriticAgentConfig,
    FactorizedAction,
    Transition,
    SimulationBuffers,
    calculate_macro_step_reward,
};
use prism_physics::molecular_dynamics::{MolecularDynamicsEngine, MolecularDynamicsConfig};
use prism_io::sovereign_types::Atom;

/// Command line arguments for Neuromorphic PRISM training
#[derive(Parser)]
#[command(name = "prism-train-neuro")]
#[command(about = "PRISM-Zero v3.1: Neuromorphic training with Flashbulb Reservoir (No PyTorch!)")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    /// Path to calibration manifest JSON file
    #[arg(short, long)]
    manifest: String,

    /// Maximum episodes per target
    #[arg(long, default_value = "1000")]
    max_episodes: usize,

    /// Output directory for checkpoints and results
    #[arg(short, long, default_value = "training_output_neuro")]
    output: String,

    /// CUDA device ID to use
    #[arg(long, default_value = "0")]
    device: usize,

    /// Checkpoint interval (episodes)
    #[arg(long, default_value = "100")]
    checkpoint_interval: usize,

    /// Early stopping patience (episodes)
    #[arg(long, default_value = "50")]
    patience: usize,

    /// Target reward for early stopping
    #[arg(long)]
    target_reward: Option<f32>,

    /// Reservoir size (number of LIF neurons)
    #[arg(long, default_value = "512")]
    reservoir_size: usize,

    /// RLS forgetting factor (lambda)
    #[arg(long, default_value = "0.99")]
    lambda: f32,

    /// Number of macro-steps per episode (Dendritic: use more, shorter chunks)
    #[arg(long, default_value = "100")]
    macro_steps: usize,

    /// Steps per macro-step (Dendritic: shorter bursts for more learning signal)
    #[arg(long, default_value = "10000")]
    steps_per_macro: u64,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

/// Training statistics
#[derive(Default)]
struct TrainingStats {
    targets_completed: usize,
    total_episodes: usize,
    total_steps: u64,
    total_transitions: usize,
    best_reward: f32,
    sum_best_rewards: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .format_module_path(false)
        .init();

    // Print header
    println!();
    println!("🧠 PRISM-Zero v{} NEUROMORPHIC Training Engine", prism_learning::PRISM_ZERO_VERSION);
    println!("⚡ Flashbulb Reservoir: E/I Balanced SNN + Reward-Modulated RLS");
    println!("🚀 NO PYTORCH REQUIRED - Pure Rust + CUDA");
    println!("{}", "═".repeat(70));
    println!();

    // Validate inputs
    if !Path::new(&args.manifest).exists() {
        anyhow::bail!("Manifest file not found: {}", args.manifest);
    }

    // Create output directory
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output directory: {}", args.output))?;

    // Load manifest
    info!("📋 Loading manifest: {}", args.manifest);
    let manifest = CalibrationManifest::load(&args.manifest)
        .context("Failed to load calibration manifest")?;

    info!("   Targets: {}", manifest.targets.len());
    info!("   Macro-steps: {} × {} steps = {}M total per episode",
          args.macro_steps, args.steps_per_macro,
          (args.macro_steps as u64 * args.steps_per_macro) / 1_000_000);

    // Create Dendritic Agent configuration
    let agent_config = DendriticAgentConfig {
        reservoir_size: args.reservoir_size,
        lambda: args.lambda,
        tau: 0.005,  // Polyak averaging coefficient
        epsilon_start: 1.0,
        epsilon_min: 0.05,
        epsilon_decay: 0.995,
        gamma: 0.99,
        target_update_freq: 100,
    };

    info!("🧬 Initializing Dendritic Agent:");
    info!("   Reservoir: {} neurons (80% E / 20% I)", agent_config.reservoir_size);
    info!("   Adaptive τ: 5-50ms (fast I, gradient E)");
    info!("   RLS lambda: {} (forgetting factor)", agent_config.lambda);
    info!("   Features: 23 raw → 46 expanded (+ velocity)");
    info!("   CUDA device: {}", args.device);

    // Create agent
    let mut agent = DendriticAgent::new_with_config(23, args.device, agent_config)
        .context("Failed to create Dendritic Agent")?;

    // Training loop
    let mut stats = TrainingStats::default();
    let training_start = Instant::now();

    for (target_idx, target) in manifest.targets.iter().enumerate() {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("🎯 Target {}/{}: {}", target_idx + 1, manifest.targets.len(), target.name);
        info!("   Family: {}, Difficulty: {}", target.family, target.difficulty);
        info!("   Target residues: {:?}", target.target_residues);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Load protein structure from PDB file
        let pdb_data = fs::read(&target.apo_pdb)
            .with_context(|| format!("Failed to read PDB: {}", target.apo_pdb))?;
        info!("   Loaded {} bytes from {}", pdb_data.len(), target.apo_pdb);

        // Reset agent for new target (clears neuronal state, keeps weights)
        agent.reset_episode()?;

        let mut best_episode_reward = f32::NEG_INFINITY;
        let mut episodes_without_improvement = 0;

        for episode in 0..args.max_episodes {
            let episode_start = Instant::now();

            // Run one episode with macro-step training
            let (episode_reward, transitions, steps) = run_macro_step_episode(
                &mut agent,
                &manifest,
                target,
                &pdb_data,
                args.macro_steps,
                args.steps_per_macro,
            )?;

            stats.total_steps += steps;
            stats.total_transitions += transitions.len();

            // Train on episode transitions (Reward-Modulated RLS)
            let batch: Vec<_> = transitions.iter()
                .map(|t| (t.state.clone(), t.action, t.reward, t.next_state.clone(), t.done))
                .collect();

            let avg_error = agent.train(batch)?;

            // Track best reward
            if episode_reward > best_episode_reward {
                best_episode_reward = episode_reward;
                episodes_without_improvement = 0;
            } else {
                episodes_without_improvement += 1;
            }

            // Logging
            let episode_time = episode_start.elapsed();
            if episode % 10 == 0 || episode == args.max_episodes - 1 {
                info!(
                    "   Episode {:4}: reward={:+.3}, best={:+.3}, ε={:.3}, RLS_err={:.4}, time={:.1}s",
                    episode, episode_reward, best_episode_reward,
                    agent.get_epsilon(), avg_error, episode_time.as_secs_f32()
                );
            }

            stats.total_episodes += 1;

            // Early stopping on target reward
            if let Some(target_reward) = args.target_reward {
                if best_episode_reward >= target_reward {
                    info!("🎉 Target reward reached: {:.3} >= {:.3}", best_episode_reward, target_reward);
                    break;
                }
            }

            // Early stopping on patience
            if episodes_without_improvement >= args.patience {
                info!("⏹️  Early stopping: {} episodes without improvement", args.patience);
                break;
            }

            // Checkpoint
            if episode > 0 && episode % args.checkpoint_interval == 0 {
                let checkpoint_path = format!("{}/checkpoint_{}_{}.json", args.output, target.name, episode);
                agent.save(&checkpoint_path)?;
                info!("💾 Checkpoint saved: {}", checkpoint_path);
            }
        }

        stats.targets_completed += 1;
        stats.sum_best_rewards += best_episode_reward;
        if best_episode_reward > stats.best_reward {
            stats.best_reward = best_episode_reward;
        }

        info!("✅ Target {} completed: best_reward={:+.3}", target.name, best_episode_reward);

        // Save per-target checkpoint
        let target_path = format!("{}/agent_after_{}.json", args.output, target.name);
        agent.save(&target_path)?;
    }

    // Final summary
    let training_time = training_start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("✅ NEUROMORPHIC TRAINING COMPLETED");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("📊 Statistics:");
    println!("   Targets completed: {}", stats.targets_completed);
    println!("   Total episodes: {}", stats.total_episodes);
    println!("   Total simulation steps: {} ({:.1}B)", stats.total_steps, stats.total_steps as f64 / 1e9);
    println!("   Total transitions: {}", stats.total_transitions);
    println!("   Average best reward: {:+.3}", stats.sum_best_rewards / stats.targets_completed.max(1) as f32);
    println!("   Best overall reward: {:+.3}", stats.best_reward);
    println!("   Training time: {:.1} minutes", training_time.as_secs_f32() / 60.0);
    println!();
    println!("📁 Results saved to: {}/", args.output);
    println!();

    // Save final agent
    let final_path = format!("{}/dendritic_agent_final.json", args.output);
    agent.save(&final_path)?;
    info!("💾 Final agent saved: {}", final_path);

    Ok(())
}

/// Run a single episode with macro-step chunking
///
/// This is the key innovation: instead of 1M steps → 1 transition,
/// we do 100 × 10K steps → 100 transitions. More learning signal!
fn run_macro_step_episode(
    agent: &mut DendriticAgent,
    manifest: &CalibrationManifest,
    target: &prism_learning::ProteinTarget,
    pdb_data: &[u8],
    num_macro_steps: usize,
    steps_per_macro: u64,
) -> Result<(f32, Vec<Transition>, u64)> {
    let reward_weights = &manifest.training_parameters.reward_weighting;
    let feature_config = &manifest.feature_config;

    // 1. Configure MD engine
    let md_config = MolecularDynamicsConfig {
        max_steps: steps_per_macro * num_macro_steps as u64,
        dt: 0.002,          // 2 femtoseconds
        friction: 1.0,      // Default, action will modify
        temp_start: 300.0,  // Default, action will modify
        temp_end: 150.0,
        annealing_steps: steps_per_macro / 2,
        cutoff_dist: 10.0,
        spring_k: 500.0,
        bias_strength: 0.0,
        target_mode: 7,
        use_gpu: true,
        max_trajectory_memory: 256 * 1024 * 1024,
        max_workspace_memory: 128 * 1024 * 1024,
    };

    // 2. Initialize MD engine from PDB data
    let mut engine = MolecularDynamicsEngine::from_sovereign_buffer(md_config, pdb_data)
        .context("Failed to initialize MD engine")?;

    // 3. Get initial atoms and create feature extractor
    let initial_atoms = engine.get_initial_atoms().to_vec();
    let mut feature_extractor = FeatureExtractor::new(feature_config.clone(), target);
    feature_extractor.initialize(&initial_atoms);

    // 4. Create SimulationBuffers for reward calculation
    let initial_buffers = SimulationBuffers::from_atoms(&initial_atoms);
    let mut current_buffers = initial_buffers.clone();

    // 5. Extract initial features
    let initial_features = feature_extractor.extract(&initial_atoms, None);
    let mut current_features = initial_features.as_slice().to_vec();

    // 6. Select initial action
    let mut current_action = agent.select_action(&current_features);

    // 7. MACRO-STEP LOOP - The heart of the training
    let mut transitions: Vec<Transition> = Vec::new();
    let mut cumulative_reward = 0.0f32;
    let mut total_steps = 0u64;

    for macro_step in 0..num_macro_steps {
        // A. Run physics simulation for this chunk
        engine.run_nlnm_breathing(steps_per_macro)
            .context("Physics simulation failed")?;
        total_steps += steps_per_macro;

        // B. Get current atom positions from GPU
        let current_atoms = engine.get_current_atoms()
            .context("Failed to get current atoms")?;

        // C. Update SimulationBuffers with new positions
        for (i, atom) in current_atoms.iter().enumerate() {
            if i < current_buffers.num_atoms {
                let base_idx = i * 4;
                current_buffers.positions[base_idx] = atom.coords[0];
                current_buffers.positions[base_idx + 1] = atom.coords[1];
                current_buffers.positions[base_idx + 2] = atom.coords[2];
            }
        }
        current_buffers.global_step = (macro_step as u64 + 1) * steps_per_macro;

        // D. Calculate macro-step reward
        let step_reward = calculate_macro_step_reward(
            &initial_buffers,
            &current_buffers,
            &target.target_residues,
            reward_weights,
            macro_step,
            num_macro_steps,
        );
        cumulative_reward += step_reward;

        // E. Extract new features
        let next_features_vec = feature_extractor.extract(&current_atoms, None);
        let next_features = next_features_vec.as_slice().to_vec();

        // F. Select next action (while we have current state)
        let next_action = agent.select_action(&next_features);

        // G. Store transition
        let done = macro_step == num_macro_steps - 1;
        transitions.push(Transition {
            state: current_features.clone(),
            action: current_action,
            reward: step_reward,
            next_state: next_features.clone(),
            done,
        });

        // H. Advance state
        current_features = next_features;
        current_action = next_action;
    }

    Ok((cumulative_reward, transitions, total_steps))
}
