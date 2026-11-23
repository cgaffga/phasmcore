// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Disk spill of the Ghost encoder's keyed order when memory is tight
//! (Stage 2 of "Large photos with less memory"; `docs/design/image/
//! ghost-embedding-v3.md`, "Memory").
//!
//! The keyed order (the Fisher–Yates permutation of every luma AC position,
//! 4 B each: ~0.6 GB at 150 MP) is written to a file once it is generated
//! and read back in lane order, a chunk at a time, by the memory ladder's
//! lowest rung ([`crate::stego::memory::GhostRung::SpilledOrder`]). The rung
//! exists only when the caller has set a spill directory ([`set_spill_dir`])
//! with room for the file; without one (WASM, the CLI, the crate by default)
//! everything stays in memory, as before.
//!
//! The file holds key-derived data only — the order of the coefficient
//! positions, derived from the passphrase — never the message, the
//! passphrase or the image. It is created owner-only in the spill directory
//! and unlinked right after it is opened: the open descriptor serves the
//! writes and reads, no name is left on disk whatever happens to the
//! process, and the space is freed when the encode drops it (success, error
//! or cancel) or the process ends.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::stego::error::StegoError;

/// Free bytes the process may write on the volume of a directory (the
/// bridges pass `statvfs`'s `f_bavail · f_frsize`); `None` when unknown.
pub type FreeSpace = fn(&Path) -> Option<u64>;

struct SpillDir {
    path: PathBuf,
    free: FreeSpace,
}

static DIR: Mutex<Option<SpillDir>> = Mutex::new(None);

fn dir() -> std::sync::MutexGuard<'static, Option<SpillDir>> {
    DIR.lock().unwrap_or_else(|p| p.into_inner())
}

/// Set (or, with `None` or an empty path, clear) the directory the encoder
/// may spill its keyed order to, and how to read its free space. An
/// app-private temporary directory; the encoder creates it if needed.
pub fn set_spill_dir(path: Option<PathBuf>, free: FreeSpace) {
    *dir() = path.filter(|p| !p.as_os_str().is_empty()).map(|path| SpillDir { path, free });
}

/// The spill directory set, if any.
pub fn spill_dir() -> Option<PathBuf> {
    dir().as_ref().map(|d| d.path.clone())
}

/// Free space a spill file of `bytes` needs: the file, a quarter more and
/// 64 MiB (the space other writers may take meanwhile).
pub fn spill_room_needed(bytes: u64) -> u64 {
    bytes.saturating_add(bytes / 4).saturating_add(64 << 20)
}

/// Whether a spill file of `bytes` can be written now: a spill directory is
/// set, the platform supports it (Unix: the file is unlinked while open)
/// and its volume has [`spill_room_needed`] free.
pub fn spill_room(bytes: u64) -> bool {
    if !cfg!(unix) {
        return false;
    }
    let guard = dir();
    let Some(d) = guard.as_ref() else { return false };
    // Measure on the directory, or its parent before the first spill made it.
    let probe = if d.path.is_dir() { d.path.as_path() } else { d.path.parent().unwrap_or(&d.path) };
    (d.free)(probe).is_some_and(|free| free >= spill_room_needed(bytes))
}

/// Bytes per write of [`OrderSpill::write`] (a cancel is seen between them).
#[cfg(unix)]
const WRITE_CHUNK: usize = 1 << 20;

/// The keyed order on disk: `len` positions, 4 B each (little-endian u32),
/// read back with positioned reads (any thread, no shared cursor). A read
/// that fails marks the spill failed ([`Self::failed`]) and hands out zeros:
/// the encode finishes the trellis it is in and then fails with
/// `ImageTooLarge`, never a panic.
pub(crate) struct OrderSpill {
    #[cfg(unix)]
    file: std::fs::File,
    len: usize,
    failed: AtomicBool,
}

impl OrderSpill {
    /// Write `order` to a new unlinked file in the spill directory.
    /// `ImageTooLarge` if no directory is set or the file cannot be written
    /// (no space, I/O error); `Cancelled` on a cancel between chunks.
    pub(crate) fn write(order: &[u32]) -> Result<Self, StegoError> {
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dir = spill_dir().ok_or(StegoError::ImageTooLarge)?;
            std::fs::create_dir_all(&dir).map_err(|_| StegoError::ImageTooLarge)?;
            let path = dir.join(format!("order-{}-{}.bin", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
                .map_err(|_| StegoError::ImageTooLarge)?;
            // Unlinked at once: the descriptor keeps the file; nothing named
            // is left if the process ends at any point from here on.
            let _ = std::fs::remove_file(&path);
            let mut bytes = Vec::with_capacity(WRITE_CHUNK);
            for chunk in order.chunks(WRITE_CHUNK / 4) {
                crate::stego::progress::check_cancelled()?;
                bytes.clear();
                for &fi in chunk {
                    bytes.extend_from_slice(&fi.to_le_bytes());
                }
                file.write_all(&bytes).map_err(|_| StegoError::ImageTooLarge)?;
            }
            // Clean pages: the kernel can drop them under memory pressure
            // without writing them first.
            file.sync_data().map_err(|_| StegoError::ImageTooLarge)?;
            Ok(Self { file, len: order.len(), failed: AtomicBool::new(false) })
        }
        #[cfg(not(unix))]
        {
            let _ = order;
            Err(StegoError::ImageTooLarge)
        }
    }

    /// Positions in the order.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Whether a read failed (the encode must not use its result).
    pub(crate) fn failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    /// Positions `start .. start + out.len()` into `out`; on an I/O error
    /// zeros and [`Self::failed`].
    pub(crate) fn read(&self, start: usize, out: &mut [u32]) {
        if !self.read_into(start, out) {
            out.fill(0);
            self.failed.store(true, Ordering::Relaxed);
        }
    }

    fn read_into(&self, start: usize, out: &mut [u32]) -> bool {
        #[cfg(test)]
        if tests::FAIL_READS.load(Ordering::Relaxed) {
            return false;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            if start.checked_add(out.len()).is_none_or(|end| end > self.len) {
                return false;
            }
            let mut bytes = [0u8; 4 * READ_CHUNK];
            let mut done = 0;
            while done < out.len() {
                let k = (out.len() - done).min(READ_CHUNK);
                let buf = &mut bytes[..4 * k];
                if self.file.read_exact_at(buf, 4 * (start + done) as u64).is_err() {
                    return false;
                }
                for (o, b) in out[done..done + k].iter_mut().zip(buf.chunks_exact(4)) {
                    *o = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                done += k;
            }
            true
        }
        #[cfg(not(unix))]
        {
            let _ = (start, out);
            false
        }
    }

    /// Positions `range` as a vector (a cost group's, for its gather).
    pub(crate) fn read_vec(&self, range: std::ops::Range<usize>) -> Result<Vec<u32>, StegoError> {
        let mut v = vec![0u32; range.len()];
        if !self.read_into(range.start, &mut v) {
            self.failed.store(true, Ordering::Relaxed);
            return Err(StegoError::ImageTooLarge);
        }
        Ok(v)
    }
}

/// Positions per positioned read (16 KiB, on the stack).
pub(crate) const READ_CHUNK: usize = 1 << 12;

/// A window of the spilled order for reads at increasing positions (a
/// lane's changes, the assembly): one [`READ_CHUNK`] read per window.
pub(crate) struct OrderCursor<'a> {
    spill: &'a OrderSpill,
    start: usize,
    len: usize,
    buf: Box<[u32; READ_CHUNK]>,
}

impl<'a> OrderCursor<'a> {
    pub(crate) fn new(spill: &'a OrderSpill) -> Self {
        Self { spill, start: 0, len: 0, buf: Box::new([0; READ_CHUNK]) }
    }

    /// Position `k` of the order (fastest for increasing `k`).
    pub(crate) fn get(&mut self, k: usize) -> u32 {
        if k < self.start || k >= self.start + self.len {
            self.start = k;
            self.len = READ_CHUNK.min(self.spill.len().saturating_sub(k)).max(1);
            let len = self.len;
            self.spill.read(k, &mut self.buf[..len]);
        }
        self.buf[k - self.start]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Fault injection: every read fails while set (tests holding the
    /// memory budget's lock).
    pub(crate) static FAIL_READS: AtomicBool = AtomicBool::new(false);

    /// Free space for tests: plenty.
    pub(crate) fn plenty(_: &Path) -> Option<u64> {
        Some(u64::MAX / 2)
    }

    /// A fresh spill directory under `$PHASM_TMP` (or the system's temporary
    /// directory). Tests that set the process-wide directory hold the memory
    /// budget's lock (`memory::budget_test_lock`).
    pub(crate) fn test_dir(name: &str) -> PathBuf {
        let base = std::env::var_os("PHASM_TMP").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let d = base.join(format!("phasm-spill-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[cfg(unix)]
    #[test]
    fn order_round_trips_and_leaves_no_file() {
        let _guard = crate::stego::memory::budget_test_lock();
        let d = test_dir("rt");
        set_spill_dir(Some(d.clone()), plenty);
        let order: Vec<u32> = (0..100_003u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        let s = OrderSpill::write(&order).unwrap();
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "unlinked at once");
        let mut out = vec![0u32; 5000];
        s.read(97_000, &mut out[..3003]);
        assert_eq!(&out[..3003], &order[97_000..]);
        assert_eq!(s.read_vec(10..20_000).unwrap(), &order[10..20_000]);
        let mut c = OrderCursor::new(&s);
        for k in (0..order.len()).step_by(997) {
            assert_eq!(c.get(k), order[k]);
        }
        // Reads and windows across the READ_CHUNK boundaries, forwards and
        // back (a new window), up to the last position.
        let mut out = vec![0u32; 3 * READ_CHUNK + 17];
        s.read(READ_CHUNK - 9, &mut out);
        assert_eq!(&out[..], &order[READ_CHUNK - 9..4 * READ_CHUNK + 8]);
        let mut c = OrderCursor::new(&s);
        let n = order.len();
        for k in [0, READ_CHUNK - 1, READ_CHUNK, READ_CHUNK + 1, 2 * READ_CHUNK - 1, 2 * READ_CHUNK, 5, n - READ_CHUNK - 1, n - 2, n - 1] {
            assert_eq!(c.get(k), order[k], "position {k}");
        }
        assert!(!s.failed());
        // Past the end: zeros and failed, no panic.
        s.read(100_000, &mut out[..10]);
        assert!(s.failed() && out[..10].iter().all(|&v| v == 0));
        set_spill_dir(None, plenty);
        assert!(!spill_room(1));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn room_needs_a_dir_and_space() {
        let _guard = crate::stego::memory::budget_test_lock();
        fn tight(_: &Path) -> Option<u64> {
            Some(100 << 20)
        }
        fn unknown(_: &Path) -> Option<u64> {
            None
        }
        set_spill_dir(None, plenty);
        assert!(!spill_room(1));
        let d = test_dir("room");
        set_spill_dir(Some(d.clone()), tight);
        assert_eq!(spill_room(16 << 20), cfg!(unix));
        assert!(!spill_room(40 << 20));
        set_spill_dir(Some(d.clone()), unknown);
        assert!(!spill_room(1));
        set_spill_dir(Some(PathBuf::new()), plenty);
        assert!(spill_dir().is_none(), "an empty path clears it");
        set_spill_dir(None, plenty);
    }
}
