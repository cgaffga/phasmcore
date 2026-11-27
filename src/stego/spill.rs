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
//!
//! The lowest rung, the streamed encode (Stage 3,
//! [`crate::stego::memory::GhostRung::Streamed`], `ghost::stream`), keeps
//! more in such files ([`SpillFile`], [`SpillWriter`]): per position its
//! cover cost, value, pair and wet bits and side-information code, in raster
//! and lane order. These are numbers derived from the cover photo and the
//! passphrase (the photo's luma coefficient values, costs and side
//! information among them), never the message or the passphrase; the same
//! creation, unlinking and lifetime.

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

/// A new file in the spill directory, owner-only, already unlinked: the
/// descriptor keeps it; nothing named is left if the process ends at any
/// point from here on, and its space is freed when the descriptor closes.
#[cfg(unix)]
fn create_unlinked(tag: &str) -> Result<std::fs::File, StegoError> {
    use std::os::unix::fs::OpenOptionsExt;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = spill_dir().ok_or(StegoError::ImageTooLarge)?;
    std::fs::create_dir_all(&dir).map_err(|_| StegoError::ImageTooLarge)?;
    let path = dir.join(format!("{tag}-{}-{}.bin", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|_| StegoError::ImageTooLarge)?;
    let _ = std::fs::remove_file(&path);
    Ok(file)
}

/// Bytes written to a [`SpillFile`] between two `sync_data` calls
/// ([`SpillWriter`], the distribution): at most this much of a file is dirty
/// in the page cache at once, so the kernel can drop the rest under memory
/// pressure without writing it first.
pub(crate) const SYNC_BYTES: u64 = 64 << 20;

/// A spill file of the streamed encode (Stage 3, `ghost::stream`): an
/// unlinked owner-only file in the spill directory ([`create_unlinked`])
/// holding numbers derived from the cover photo and the passphrase —
/// per-position costs, coefficient values, pair and wet bits, side
/// information, positions — never the message or the passphrase. Positioned reads and writes at byte offsets, from any
/// thread. Every I/O error is `ImageTooLarge`; a read that fails also marks
/// the file failed ([`Self::failed`]).
pub(crate) struct SpillFile {
    #[cfg(unix)]
    file: std::fs::File,
    failed: AtomicBool,
}

impl SpillFile {
    /// A new empty file (`tag` names it for the instant before it is unlinked).
    pub(crate) fn create(tag: &str) -> Result<Self, StegoError> {
        #[cfg(unix)]
        {
            Ok(Self { file: create_unlinked(tag)?, failed: AtomicBool::new(false) })
        }
        #[cfg(not(unix))]
        {
            let _ = tag;
            Err(StegoError::ImageTooLarge)
        }
    }

    /// Write `bytes` at byte `off`.
    pub(crate) fn write_at(&self, off: u64, bytes: &[u8]) -> Result<(), StegoError> {
        #[cfg(test)]
        if tests::write_fault() {
            return Err(StegoError::ImageTooLarge);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.write_all_at(bytes, off).map_err(|_| StegoError::ImageTooLarge)
        }
        #[cfg(not(unix))]
        {
            let _ = (off, bytes);
            Err(StegoError::ImageTooLarge)
        }
    }

    /// Read `out.len()` bytes at byte `off`; an error marks the file failed.
    pub(crate) fn read_at(&self, off: u64, out: &mut [u8]) -> Result<(), StegoError> {
        if self.read_into(off, out) {
            Ok(())
        } else {
            self.failed.store(true, Ordering::Relaxed);
            Err(StegoError::ImageTooLarge)
        }
    }

    /// [`Self::read_at`] that hands out zeros on an error (the trellis then
    /// finishes on them and the encode fails: [`Self::failed`]).
    pub(crate) fn read_or_zeros(&self, off: u64, out: &mut [u8]) {
        if self.read_at(off, out).is_err() {
            out.fill(0);
        }
    }

    fn read_into(&self, off: u64, out: &mut [u8]) -> bool {
        #[cfg(test)]
        if tests::read_fault() {
            return false;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(out, off).is_ok()
        }
        #[cfg(not(unix))]
        {
            let _ = (off, out);
            false
        }
    }

    /// Write the file's dirty pages (clean pages can be dropped under memory
    /// pressure without being written first).
    pub(crate) fn sync(&self) -> Result<(), StegoError> {
        #[cfg(unix)]
        {
            self.file.sync_data().map_err(|_| StegoError::ImageTooLarge)
        }
        #[cfg(not(unix))]
        {
            Err(StegoError::ImageTooLarge)
        }
    }

    /// Whether a read failed (the encode must not use what it computed).
    pub(crate) fn failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }
}

/// Bytes a [`SpillWriter`] gathers per write.
pub(crate) const WRITER_BYTES: usize = 1 << 20;

/// A [`SpillFile`] written front to back in [`WRITER_BYTES`] writes, synced
/// every [`SYNC_BYTES`] and at the end.
pub(crate) struct SpillWriter {
    file: SpillFile,
    buf: Vec<u8>,
    off: u64,
    unsynced: u64,
}

impl SpillWriter {
    pub(crate) fn create(tag: &str) -> Result<Self, StegoError> {
        Ok(Self { file: SpillFile::create(tag)?, buf: Vec::with_capacity(WRITER_BYTES), off: 0, unsynced: 0 })
    }

    /// Append `bytes`.
    pub(crate) fn push(&mut self, mut bytes: &[u8]) -> Result<(), StegoError> {
        if self.buf.is_empty() && bytes.len() >= WRITER_BYTES {
            // A large piece goes to the file as it is.
            self.file.write_at(self.off, bytes)?;
            self.off += bytes.len() as u64;
            self.unsynced += bytes.len() as u64;
            if self.unsynced >= SYNC_BYTES {
                self.file.sync()?;
                self.unsynced = 0;
            }
            return Ok(());
        }
        while !bytes.is_empty() {
            let k = (WRITER_BYTES - self.buf.len()).min(bytes.len());
            self.buf.extend_from_slice(&bytes[..k]);
            bytes = &bytes[k..];
            if self.buf.len() == WRITER_BYTES {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), StegoError> {
        self.file.write_at(self.off, &self.buf)?;
        self.off += self.buf.len() as u64;
        self.unsynced += self.buf.len() as u64;
        self.buf.clear();
        if self.unsynced >= SYNC_BYTES {
            self.file.sync()?;
            self.unsynced = 0;
        }
        Ok(())
    }

    /// The written file, synced.
    pub(crate) fn finish(mut self) -> Result<SpillFile, StegoError> {
        if !self.buf.is_empty() {
            self.flush()?;
        }
        self.file.sync()?;
        Ok(self.file)
    }
}

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
            // Unlinked at once: the descriptor keeps the file; nothing named
            // is left if the process ends at any point from here on.
            let mut file = create_unlinked("order")?;
            let mut bytes = Vec::with_capacity(WRITE_CHUNK);
            for chunk in order.chunks(WRITE_CHUNK / 4) {
                crate::stego::progress::check_cancelled()?;
                bytes.clear();
                for &fi in chunk {
                    bytes.extend_from_slice(&fi.to_le_bytes());
                }
                #[cfg(test)]
                if tests::write_fault() {
                    return Err(StegoError::ImageTooLarge);
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
        if tests::read_fault() {
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

    /// Positions `start .. start + out.len()` into `out`; `ImageTooLarge` on
    /// an I/O error (the streamed encode's inverse order).
    pub(crate) fn read_checked(&self, start: usize, out: &mut [u32]) -> Result<(), StegoError> {
        if self.read_into(start, out) {
            Ok(())
        } else {
            self.failed.store(true, Ordering::Relaxed);
            Err(StegoError::ImageTooLarge)
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

    /// Fault injection: every write fails while set (a full disk).
    pub(crate) static FAIL_WRITES: AtomicBool = AtomicBool::new(false);

    /// Fault injection from the n-th read / write on (0: off): the n-th and
    /// every later one fail (sets [`FAIL_READS`] / [`FAIL_WRITES`]).
    pub(crate) static FAIL_READ_AT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    pub(crate) static FAIL_WRITE_AT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn fault(all: &AtomicBool, at: &std::sync::atomic::AtomicUsize) -> bool {
        if all.load(Ordering::Relaxed) {
            return true;
        }
        let hit = at.try_update(Ordering::Relaxed, Ordering::Relaxed, |k| if k > 1 { Some(k - 1) } else if k == 1 { Some(0) } else { None });
        if hit == Ok(1) {
            all.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }

    pub(crate) fn read_fault() -> bool {
        fault(&FAIL_READS, &FAIL_READ_AT)
    }

    pub(crate) fn write_fault() -> bool {
        fault(&FAIL_WRITES, &FAIL_WRITE_AT)
    }

    /// Clear every fault injection.
    pub(crate) fn clear_faults() {
        FAIL_READ_AT.store(0, Ordering::Relaxed);
        FAIL_WRITE_AT.store(0, Ordering::Relaxed);
        FAIL_READS.store(false, Ordering::Relaxed);
        FAIL_WRITES.store(false, Ordering::Relaxed);
    }

    /// A spill file written in pieces of every size reads back as written,
    /// at any offset; errors are clean and nothing named is left.
    #[cfg(unix)]
    #[test]
    fn spill_file_round_trips() {
        let _guard = crate::stego::memory::budget_test_lock();
        let d = test_dir("file");
        set_spill_dir(Some(d.clone()), plenty);
        let data: Vec<u8> = (0..3 * WRITER_BYTES + 12_345).map(|i| (i * 7 + i / 251) as u8).collect();
        let mut w = SpillWriter::create("t").unwrap();
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "unlinked at once");
        let mut at = 0;
        for k in [1usize, 17, WRITER_BYTES - 3, 4096, 2 * WRITER_BYTES + 5] {
            let end = (at + k).min(data.len());
            w.push(&data[at..end]).unwrap();
            at = end;
        }
        w.push(&data[at..]).unwrap();
        let f = w.finish().unwrap();
        let mut out = vec![0u8; 70_000];
        for off in [0usize, 1, WRITER_BYTES - 5, data.len() - 70_000] {
            f.read_at(off as u64, &mut out).unwrap();
            assert_eq!(out, data[off..off + 70_000], "offset {off}");
        }
        // Positioned writes anywhere.
        f.write_at(10, &[1, 2, 3]).unwrap();
        f.read_at(9, &mut out[..5]).unwrap();
        assert_eq!(out[..5], [data[9], 1, 2, 3, data[13]]);
        assert!(!f.failed());
        // Past the end: an error, the file marked failed; zeros from the
        // zero-filling read.
        assert!(f.read_at(data.len() as u64 - 2, &mut out[..4]).is_err());
        assert!(f.failed());
        f.read_or_zeros(data.len() as u64, &mut out[..4]);
        assert_eq!(out[..4], [0; 4]);
        // A failing write (full disk): a clean error.
        FAIL_WRITES.store(true, Ordering::Relaxed);
        assert!(matches!(f.write_at(0, &[1]), Err(StegoError::ImageTooLarge)));
        let mut w = SpillWriter::create("t").unwrap();
        assert!(w.push(&data[..WRITER_BYTES]).is_err());
        assert!(OrderSpill::write(&[1, 2, 3]).is_err());
        FAIL_WRITES.store(false, Ordering::Relaxed);
        drop((f, w));
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
        set_spill_dir(None, plenty);
        assert!(SpillFile::create("t").is_err(), "no directory");
        let _ = std::fs::remove_dir_all(&d);
    }

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
