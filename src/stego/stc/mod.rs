pub mod hhat;
pub mod embed;
pub mod extract;

/// Parameters for the STC codec.
pub struct StcParams {
    /// Submatrix height (constraint length). h=7 → 128 Viterbi states.
    pub h: usize,
    /// Submatrix width: ceil(n / m) where n = cover length, m = message length.
    pub w: usize,
}
