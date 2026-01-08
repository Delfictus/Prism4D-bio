//! Target-Aware Feature Extraction for Cryptic Site Prediction
//!
//! This module implements VRAM-efficient, target-aware feature extraction that enables
//! the RL agent to generalize across diverse protein targets.
//!
//! ## Feature Categories (23 total dimensions by default)
//!
//! | Category | Dims | Description |
//! |----------|------|-------------|
//! | Global | 3 | Size, Radius of Gyration, Density |
//! | Target Neighborhood | 8 | Target-specific exposure and contacts |
//! | Stability | 4 | RMSD, clashes, max displacement |
//! | Family Flags | 4 | Multimer, glycan, size class |
//! | Temporal | 4 | Change from initial state |
//!
//! ## Design Principles
//! - O(N) complexity using spatial hashing
//! - No large matrix allocations (VRAM-safe)
//! - Invariant to protein identity (generalizable)
//! - Computed on CPU with simple reductions

use crate::buffers::SimulationBuffers;
use crate::manifest::{FeatureConfig, ProteinTarget};
use prism_io::sovereign_types::Atom;
use std::collections::{HashMap, HashSet};

// ============================================================================
// FEATURE VECTOR
// ============================================================================

/// Complete feature vector for RL agent input
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Raw feature values
    pub values: Vec<f32>,
    /// Feature dimension
    pub dim: usize,
}

impl FeatureVector {
    /// Create new feature vector with given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
            dim,
        }
    }

    /// Get slice for agent input
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Clone into owned Vec
    pub fn to_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
}

// ============================================================================
// FEATURE EXTRACTOR
// ============================================================================

/// Target-aware feature extractor
///
/// Maintains reference to initial state for temporal features
pub struct FeatureExtractor {
    config: FeatureConfig,
    target_residues: HashSet<usize>,
    core_residues: HashSet<usize>,
    is_multimer: bool,
    has_glycans: bool,
    initial_exposure: f32,
    initial_rg: f32,
}

impl FeatureExtractor {
    /// Create new feature extractor for a target
    pub fn new(config: FeatureConfig, target: &ProteinTarget) -> Self {
        Self {
            config,
            target_residues: target.target_residues.iter().cloned().collect(),
            core_residues: target.core_residues.iter().cloned().collect(),
            is_multimer: target.is_multimer,
            has_glycans: target.has_glycans,
            initial_exposure: 0.0,
            initial_rg: 0.0,
        }
    }

    /// Initialize with reference to initial state
    pub fn initialize(&mut self, initial_atoms: &[Atom]) {
        if !initial_atoms.is_empty() {
            let positions = atoms_to_flat_positions(initial_atoms);
            self.initial_rg = calculate_radius_of_gyration(&positions);

            // Calculate initial target exposure
            let target_indices: Vec<usize> = initial_atoms.iter()
                .enumerate()
                .filter(|(_, a)| self.target_residues.contains(&(a.residue_id as usize)))
                .map(|(i, _)| i)
                .collect();

            self.initial_exposure = calculate_exposure_fast(&positions, &target_indices, self.config.neighbor_cutoff);
        }
    }

    /// Extract complete feature vector from current state
    pub fn extract(&self, atoms: &[Atom], initial_atoms: Option<&[Atom]>) -> FeatureVector {
        let mut features = FeatureVector::new(self.config.feature_dim());
        let mut idx = 0;

        let positions = atoms_to_flat_positions(atoms);
        let n = atoms.len() as f32;

        // Identify target and core atom indices
        let (target_indices, core_indices) = self.identify_atom_groups(atoms);

        // 1. Global Features (3)
        if self.config.include_global {
            let (size, rg, density) = self.extract_global(&positions, n);
            features.values[idx] = size;
            features.values[idx + 1] = rg;
            features.values[idx + 2] = density;
            idx += 3;
        }

        // 2. Target Neighborhood Features (8)
        if self.config.include_target_neighborhood {
            let neighborhood = self.extract_target_neighborhood(&positions, &target_indices, &core_indices);
            for v in neighborhood {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 3. Stability Features (4)
        if self.config.include_stability {
            let initial_pos = initial_atoms.map(|a| atoms_to_flat_positions(a));
            let stability = self.extract_stability(&positions, initial_pos.as_ref(), &core_indices);
            for v in stability {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 4. Family Flags (4)
        if self.config.include_family_flags {
            let flags = self.extract_family_flags(n);
            for v in flags {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 5. Temporal Features (4)
        if self.config.include_temporal {
            let temporal = self.extract_temporal(&positions, &target_indices);
            for v in temporal {
                features.values[idx] = v;
                idx += 1;
            }
        }

        features
    }

    /// Identify target and core atom indices
    fn identify_atom_groups(&self, atoms: &[Atom]) -> (Vec<usize>, Vec<usize>) {
        let mut target_indices = Vec::new();
        let mut core_indices = Vec::new();

        for (i, atom) in atoms.iter().enumerate() {
            let res_id = atom.residue_id as usize;
            if self.target_residues.contains(&res_id) {
                target_indices.push(i);
            } else if self.core_residues.is_empty() || self.core_residues.contains(&res_id) {
                core_indices.push(i);
            }
        }

        (target_indices, core_indices)
    }

    // ========================================================================
    // GLOBAL FEATURES (3 dims)
    // ========================================================================

    fn extract_global(&self, positions: &[f32], n: f32) -> (f32, f32, f32) {
        // Size: normalized atom count (0-1 for typical proteins)
        let size = (n / 10000.0).min(1.0);

        // Radius of Gyration
        let rg = calculate_radius_of_gyration(positions);

        // Density: atoms per unit volume (normalized)
        let density = if rg > 0.0 { n / (rg.powi(3) + 1e-6) } else { 0.0 };
        let density_norm = (density / 100.0).min(1.0);

        (size, rg / 50.0, density_norm) // Normalize Rg to ~[0,1]
    }

    // ========================================================================
    // TARGET NEIGHBORHOOD FEATURES (8 dims)
    // ========================================================================

    fn extract_target_neighborhood(
        &self,
        positions: &[f32],
        target_indices: &[usize],
        _core_indices: &[usize],
    ) -> [f32; 8] {
        let cutoff = self.config.neighbor_cutoff;
        let contact_cutoff = self.config.contact_cutoff;

        if target_indices.is_empty() {
            return [0.0; 8];
        }

        let grid = SpatialGrid::new(positions, cutoff);

        // 1. Mean exposure of target residues (0 = buried, 1 = exposed)
        let mut total_exposure = 0.0;
        let mut min_exposure = f32::MAX;
        let mut max_exposure = 0.0f32;

        // 2. Contact counts
        let mut total_contacts = 0.0;
        let mut min_contacts = f32::MAX;
        let mut max_contacts = 0.0f32;

        // 3. Distance to nearest non-target
        let mut min_dist_to_non_target = f32::MAX;
        let mut mean_dist_to_non_target = 0.0;

        let target_set: HashSet<usize> = target_indices.iter().cloned().collect();

        for &idx in target_indices {
            let (neighbors, non_target_dist) = grid.count_neighbors_detailed(
                positions, idx, &target_set, cutoff
            );

            // Exposure proxy: fewer neighbors = more exposed
            let exposure = 1.0 / (1.0 + neighbors);
            total_exposure += exposure;
            min_exposure = min_exposure.min(exposure);
            max_exposure = max_exposure.max(exposure);

            // Contact counts within tighter cutoff
            let contacts = grid.count_within_cutoff(positions, idx, contact_cutoff);
            total_contacts += contacts;
            min_contacts = min_contacts.min(contacts);
            max_contacts = max_contacts.max(contacts);

            // Distance to nearest non-target
            if non_target_dist < min_dist_to_non_target {
                min_dist_to_non_target = non_target_dist;
            }
            mean_dist_to_non_target += non_target_dist;
        }

        let n = target_indices.len() as f32;
        let mean_exposure = total_exposure / n;
        let mean_contacts = total_contacts / n;
        mean_dist_to_non_target /= n;

        [
            mean_exposure,                              // 0: Mean target exposure [0,1]
            min_exposure.min(1.0),                      // 1: Min target exposure (bottleneck)
            max_exposure.min(1.0),                      // 2: Max target exposure
            (mean_contacts / 20.0).min(1.0),            // 3: Mean contact count (normalized)
            (max_contacts / 30.0).min(1.0),             // 4: Max contacts (crowding)
            (min_dist_to_non_target / 10.0).min(1.0),   // 5: Min dist to non-target (boundary)
            (mean_dist_to_non_target / 15.0).min(1.0),  // 6: Mean dist to non-target
            (target_indices.len() as f32 / 100.0).min(1.0), // 7: Target region size
        ]
    }

    // ========================================================================
    // STABILITY FEATURES (4 dims)
    // ========================================================================

    fn extract_stability(
        &self,
        positions: &[f32],
        initial_positions: Option<&Vec<f32>>,
        core_indices: &[usize],
    ) -> [f32; 4] {
        let clash_dist_sq = (self.config.contact_cutoff * 0.25).powi(2); // ~1.5Å

        // 1. Core RMSD from initial (if available)
        let core_rmsd = if let Some(init_pos) = initial_positions {
            calculate_core_rmsd_fast(positions, init_pos, core_indices)
        } else {
            0.0
        };

        // 2. Clash count (atoms too close together)
        let clash_count = count_clashes(positions, clash_dist_sq);

        // 3. Max displacement from initial
        let max_displacement = if let Some(init_pos) = initial_positions {
            calculate_max_displacement(positions, init_pos)
        } else {
            0.0
        };

        // 4. Local RMSD variance (stability heterogeneity)
        let rmsd_variance = if let Some(init_pos) = initial_positions {
            calculate_displacement_variance(positions, init_pos)
        } else {
            0.0
        };

        [
            (core_rmsd / 5.0).min(1.0),              // Normalized RMSD
            (clash_count as f32 / 10.0).min(1.0),   // Clash severity
            (max_displacement / 10.0).min(1.0),      // Max displacement
            (rmsd_variance / 5.0).min(1.0),          // Displacement variance
        ]
    }

    // ========================================================================
    // FAMILY FLAGS (4 dims)
    // ========================================================================

    fn extract_family_flags(&self, n: f32) -> [f32; 4] {
        // 1. Is multimer flag
        let multimer_flag = if self.is_multimer { 1.0 } else { 0.0 };

        // 2. Has glycans flag
        let glycan_flag = if self.has_glycans { 1.0 } else { 0.0 };

        // 3. Size class (small/medium/large/huge)
        let size_class = if n < 1000.0 {
            0.0 // Small
        } else if n < 5000.0 {
            0.33 // Medium
        } else if n < 15000.0 {
            0.66 // Large
        } else {
            1.0 // Huge (spike-like)
        };

        // 4. Target fraction (what % of protein is target)
        let target_fraction = (self.target_residues.len() as f32 / (n / 10.0).max(1.0)).min(1.0);

        [multimer_flag, glycan_flag, size_class, target_fraction]
    }

    // ========================================================================
    // TEMPORAL FEATURES (4 dims)
    // ========================================================================

    fn extract_temporal(&self, positions: &[f32], target_indices: &[usize]) -> [f32; 4] {
        // Current exposure
        let current_exposure = calculate_exposure_fast(
            positions, target_indices, self.config.neighbor_cutoff
        );

        // Exposure change from initial
        let exposure_delta = current_exposure - self.initial_exposure;

        // Current Rg
        let current_rg = calculate_radius_of_gyration(positions);

        // Rg change from initial
        let rg_delta = current_rg - self.initial_rg;

        [
            current_exposure,                           // 0: Current exposure
            (exposure_delta + 0.5).max(0.0).min(1.0),  // 1: Exposure change (centered)
            (current_rg / 50.0).min(1.0),              // 2: Current Rg (normalized)
            ((rg_delta / 10.0) + 0.5).max(0.0).min(1.0), // 3: Rg change (centered)
        ]
    }
}

// ============================================================================
// SPATIAL GRID (O(N) Neighbor Search)
// ============================================================================

struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    fn new(positions: &[f32], cell_size: f32) -> Self {
        let mut cells = HashMap::new();
        let n = positions.len() / 3;

        for i in 0..n {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let key = (
                (x / cell_size).floor() as i32,
                (y / cell_size).floor() as i32,
                (z / cell_size).floor() as i32,
            );

            cells.entry(key).or_insert_with(Vec::new).push(i);
        }

        Self { cell_size, cells }
    }

    fn count_neighbors_detailed(
        &self,
        positions: &[f32],
        atom_idx: usize,
        target_set: &HashSet<usize>,
        cutoff: f32,
    ) -> (f32, f32) {
        let cutoff_sq = cutoff * cutoff;
        let x = positions[atom_idx * 3];
        let y = positions[atom_idx * 3 + 1];
        let z = positions[atom_idx * 3 + 2];

        let cx = (x / self.cell_size).floor() as i32;
        let cy = (y / self.cell_size).floor() as i32;
        let cz = (z / self.cell_size).floor() as i32;

        let mut count = 0.0;
        let mut min_non_target_dist = f32::MAX;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if atom_idx == j { continue; }

                            let x2 = positions[j * 3];
                            let y2 = positions[j * 3 + 1];
                            let z2 = positions[j * 3 + 2];

                            let dist_sq = (x - x2).powi(2) + (y - y2).powi(2) + (z - z2).powi(2);

                            if dist_sq < cutoff_sq {
                                count += 1.0;
                            }

                            // Track distance to non-target atoms
                            if !target_set.contains(&j) && dist_sq < min_non_target_dist {
                                min_non_target_dist = dist_sq;
                            }
                        }
                    }
                }
            }
        }

        (count, min_non_target_dist.sqrt())
    }

    fn count_within_cutoff(&self, positions: &[f32], atom_idx: usize, cutoff: f32) -> f32 {
        let cutoff_sq = cutoff * cutoff;
        let x = positions[atom_idx * 3];
        let y = positions[atom_idx * 3 + 1];
        let z = positions[atom_idx * 3 + 2];

        let cx = (x / self.cell_size).floor() as i32;
        let cy = (y / self.cell_size).floor() as i32;
        let cz = (z / self.cell_size).floor() as i32;

        let mut count = 0.0;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if atom_idx == j { continue; }

                            let x2 = positions[j * 3];
                            let y2 = positions[j * 3 + 1];
                            let z2 = positions[j * 3 + 2];

                            let dist_sq = (x - x2).powi(2) + (y - y2).powi(2) + (z - z2).powi(2);
                            if dist_sq < cutoff_sq {
                                count += 1.0;
                            }
                        }
                    }
                }
            }
        }

        count
    }
}

// ============================================================================
// HELPER FUNCTIONS (VRAM-Safe, O(N))
// ============================================================================

/// Convert atoms to flat position array [x0,y0,z0,x1,y1,z1,...]
fn atoms_to_flat_positions(atoms: &[Atom]) -> Vec<f32> {
    let mut positions = Vec::with_capacity(atoms.len() * 3);
    for atom in atoms {
        positions.push(atom.coords[0]);
        positions.push(atom.coords[1]);
        positions.push(atom.coords[2]);
    }
    positions
}

/// Calculate radius of gyration
fn calculate_radius_of_gyration(positions: &[f32]) -> f32 {
    let n = positions.len() / 3;
    if n == 0 { return 0.0; }

    // Calculate center of mass
    let mut com = [0.0f32; 3];
    for i in 0..n {
        com[0] += positions[i * 3];
        com[1] += positions[i * 3 + 1];
        com[2] += positions[i * 3 + 2];
    }
    let n_f = n as f32;
    com[0] /= n_f;
    com[1] /= n_f;
    com[2] /= n_f;

    // Calculate Rg
    let mut rg_sq = 0.0;
    for i in 0..n {
        let dx = positions[i * 3] - com[0];
        let dy = positions[i * 3 + 1] - com[1];
        let dz = positions[i * 3 + 2] - com[2];
        rg_sq += dx * dx + dy * dy + dz * dz;
    }

    (rg_sq / n_f).sqrt()
}

/// Fast exposure calculation using spatial grid
fn calculate_exposure_fast(positions: &[f32], target_indices: &[usize], cutoff: f32) -> f32 {
    if target_indices.is_empty() { return 0.0; }

    let grid = SpatialGrid::new(positions, cutoff);
    let mut total_exposure = 0.0;

    for &idx in target_indices {
        let neighbors = grid.count_within_cutoff(positions, idx, cutoff);
        total_exposure += 1.0 / (1.0 + neighbors);
    }

    total_exposure / target_indices.len() as f32
}

/// Fast core RMSD calculation
fn calculate_core_rmsd_fast(positions: &[f32], initial: &[f32], core_indices: &[usize]) -> f32 {
    if core_indices.is_empty() { return 0.0; }

    let mut sum_sq = 0.0;
    for &idx in core_indices {
        let base = idx * 3;
        if base + 2 < positions.len() && base + 2 < initial.len() {
            let dx = positions[base] - initial[base];
            let dy = positions[base + 1] - initial[base + 1];
            let dz = positions[base + 2] - initial[base + 2];
            sum_sq += dx * dx + dy * dy + dz * dz;
        }
    }

    (sum_sq / core_indices.len() as f32).sqrt()
}

/// Count atomic clashes (pairs closer than threshold)
fn count_clashes(positions: &[f32], clash_dist_sq: f32) -> usize {
    let n = positions.len() / 3;
    let mut clashes = 0;

    // Use grid for efficiency
    let grid = SpatialGrid::new(positions, clash_dist_sq.sqrt() * 2.0);

    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        let cx = (x / (clash_dist_sq.sqrt() * 2.0)).floor() as i32;
        let cy = (y / (clash_dist_sq.sqrt() * 2.0)).floor() as i32;
        let cz = (z / (clash_dist_sq.sqrt() * 2.0)).floor() as i32;

        for dx in 0..=1 {
            for dy in 0..=1 {
                for dz in 0..=1 {
                    if let Some(indices) = grid.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if j <= i { continue; }

                            let x2 = positions[j * 3];
                            let y2 = positions[j * 3 + 1];
                            let z2 = positions[j * 3 + 2];

                            let dist_sq = (x - x2).powi(2) + (y - y2).powi(2) + (z - z2).powi(2);
                            if dist_sq < clash_dist_sq {
                                clashes += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    clashes
}

/// Calculate maximum displacement from initial
fn calculate_max_displacement(positions: &[f32], initial: &[f32]) -> f32 {
    let n = positions.len().min(initial.len()) / 3;
    let mut max_disp = 0.0f32;

    for i in 0..n {
        let base = i * 3;
        let dx = positions[base] - initial[base];
        let dy = positions[base + 1] - initial[base + 1];
        let dz = positions[base + 2] - initial[base + 2];
        let disp = (dx * dx + dy * dy + dz * dz).sqrt();
        max_disp = max_disp.max(disp);
    }

    max_disp
}

/// Calculate variance in displacement (heterogeneity)
fn calculate_displacement_variance(positions: &[f32], initial: &[f32]) -> f32 {
    let n = positions.len().min(initial.len()) / 3;
    if n == 0 { return 0.0; }

    let mut displacements = Vec::with_capacity(n);
    let mut sum = 0.0;

    for i in 0..n {
        let base = i * 3;
        let dx = positions[base] - initial[base];
        let dy = positions[base + 1] - initial[base + 1];
        let dz = positions[base + 2] - initial[base + 2];
        let disp = (dx * dx + dy * dy + dz * dz).sqrt();
        displacements.push(disp);
        sum += disp;
    }

    let mean = sum / n as f32;
    let variance: f32 = displacements.iter()
        .map(|&d| (d - mean).powi(2))
        .sum::<f32>() / n as f32;

    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_dim() {
        let config = FeatureConfig::default();
        assert_eq!(config.feature_dim(), 23);
    }

    #[test]
    fn test_radius_of_gyration() {
        // Simple cube of atoms
        let positions = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
        ];
        let rg = calculate_radius_of_gyration(&positions);
        assert!(rg > 0.0 && rg < 2.0);
    }
}
