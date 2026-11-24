//! Buffered reads of GGUF headers.
//!
//! Parsing a header is hundreds of thousands of small reads: Inkling 2's
//! 200,000-token vocabulary alone is a length and a string per token, and the
//! parsers also ask for the cursor position around most fields. Straight on a
//! `File`, every one of those is a system call; on Windows one pass over that
//! header took about five seconds, against 0.25 s to build the tokenizer from
//! the same metadata in memory. [`HeaderReader`] serves them from a buffer and
//! tracks its position itself.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use candle_core::quantized::gguf_file;

const BUFFER_BYTES: usize = 256 * 1024;

/// Buffered `Read + Seek` over an open file, for header parsing. Seeks only
/// move a logical cursor; bytes come from one buffer that is refilled when a
/// read falls outside it. Call [`HeaderReader::finish`] to leave the file
/// cursor where the same reads straight on the file would have.
pub(crate) struct HeaderReader<'a> {
    file: &'a mut File,
    file_bytes: u64,
    position: u64,
    buffer: Vec<u8>,
    buffer_start: u64,
}

impl<'a> HeaderReader<'a> {
    /// Start at the file's current cursor.
    pub(crate) fn new(file: &'a mut File) -> io::Result<Self> {
        let file_bytes = file.metadata()?.len();
        let position = file.stream_position()?;
        Ok(Self {
            file,
            file_bytes,
            position,
            buffer: Vec::new(),
            buffer_start: 0,
        })
    }

    /// Length of the file when the reader was created.
    pub(crate) fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// Move the file cursor to the reader's position.
    pub(crate) fn finish(self) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(self.position)).map(drop)
    }

    fn buffered(&self) -> Option<&[u8]> {
        let offset = self.position.checked_sub(self.buffer_start)?;
        let offset = usize::try_from(offset).ok()?;
        (offset < self.buffer.len()).then(|| &self.buffer[offset..])
    }

    fn refill(&mut self) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(self.position))?;
        self.buffer.resize(BUFFER_BYTES, 0);
        let mut filled = 0;
        while filled < self.buffer.len() {
            match self.file.read(&mut self.buffer[filled..]) {
                Ok(0) => break,
                Ok(read) => filled += read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => {
                    self.buffer.clear();
                    return Err(error);
                }
            }
        }
        self.buffer.truncate(filled);
        self.buffer_start = self.position;
        Ok(())
    }
}

impl Read for HeaderReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.buffered().is_none() {
            if out.len() >= BUFFER_BYTES {
                self.file.seek(SeekFrom::Start(self.position))?;
                let read = self.file.read(out)?;
                self.position += read as u64;
                return Ok(read);
            }
            self.refill()?;
        }
        let Some(available) = self.buffered() else {
            return Ok(0);
        };
        let read = out.len().min(available.len());
        out[..read].copy_from_slice(&available[..read]);
        self.position += read as u64;
        Ok(read)
    }
}

impl Seek for HeaderReader<'_> {
    fn seek(&mut self, target: SeekFrom) -> io::Result<u64> {
        let position = match target {
            SeekFrom::Start(offset) => Some(offset),
            SeekFrom::Current(delta) => self.position.checked_add_signed(delta),
            SeekFrom::End(delta) => self.file_bytes.checked_add_signed(delta),
        };
        self.position = position.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            )
        })?;
        Ok(self.position)
    }
}

/// Candle's GGUF header reader through a [`HeaderReader`]. The file cursor
/// ends where Candle's reader leaves it, on success or failure.
pub(crate) fn read_candle_gguf(file: &mut File) -> candle_core::Result<gguf_file::Content> {
    let mut reader = HeaderReader::new(file)?;
    let content = gguf_file::Content::read(&mut reader);
    reader.finish()?;
    content
}
