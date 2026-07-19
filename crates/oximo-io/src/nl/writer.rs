//! Token-level writer abstraction.
//!
//! Wraps `std::io::Write` and a `WriteOptions` slice. All NL emission goes
//! through this type so the ASCII / binary distinction, precision control,
//! comment toggle, and NaN/Inf handling stay in one place.

use std::io::{self, Write};

use crate::error::IoError;

use super::options::{NlFormat, WriteOptions};

pub(crate) struct Writer<'a, W: Write> {
    pub(crate) out: &'a mut W,
    pub(crate) opts: &'a WriteOptions,
}

impl<'a, W: Write> Writer<'a, W> {
    pub(crate) fn new(out: &'a mut W, opts: &'a WriteOptions) -> Self {
        Self { out, opts }
    }

    /// Raw text passthrough; used for the (always-ASCII) header lines.
    pub(crate) fn write_text(&mut self, s: &str) -> io::Result<()> {
        self.out.write_all(s.as_bytes())
    }

    /// Header trailing comment helper. ASCII: writes `\t# {comment}\n`
    /// when `opts.comments`, else just `\n`. Binary: always `\n` (header is
    /// ASCII even in binary mode, but carries no comments).
    pub(crate) fn header_eol(&mut self, comment: &str) -> io::Result<()> {
        if self.opts.format == NlFormat::Ascii && self.opts.comments {
            writeln!(self.out, "\t# {comment}")
        } else {
            writeln!(self.out)
        }
    }

    /// Opcode token (`o<N>`). ASCII: `o<N>\n`. Binary: `o` byte + 4-byte LE int.
    pub(crate) fn op(&mut self, code: u32) -> io::Result<()> {
        match self.opts.format {
            NlFormat::Ascii => writeln!(self.out, "o{code}"),
            NlFormat::Binary => {
                self.out.write_all(b"o")?;
                self.out.write_all(&nl_i32(code, "opcode")?.to_le_bytes())
            }
        }
    }

    /// Variable reference token (`v<i>`).
    pub(crate) fn var(&mut self, idx: u32) -> io::Result<()> {
        match self.opts.format {
            NlFormat::Ascii => writeln!(self.out, "v{idx}"),
            NlFormat::Binary => {
                self.out.write_all(b"v")?;
                self.out.write_all(&nl_i32(idx, "variable index")?.to_le_bytes())
            }
        }
    }

    /// Numeric constant inside an expression. Binary picks `s<short>` /
    /// `l<long>` / `n<double>` based on the value's magnitude and integrality.
    pub(crate) fn num(&mut self, x: f64) -> Result<(), IoError> {
        match self.opts.format {
            NlFormat::Ascii => {
                let s = self.fmt_value(x)?;
                writeln!(self.out, "n{s}")?;
            }
            NlFormat::Binary => {
                if let Some(short) = as_short(x) {
                    self.out.write_all(b"s")?;
                    self.out.write_all(&short.to_le_bytes())?;
                } else if let Some(long) = as_long(x) {
                    self.out.write_all(b"l")?;
                    self.out.write_all(&long.to_le_bytes())?;
                } else if x.is_finite() {
                    self.out.write_all(b"n")?;
                    self.out.write_all(&x.to_le_bytes())?;
                } else if self.opts.nonfinite_strings {
                    // Binary mode has no string form for Inf/NaN; emit as a
                    // raw double, which is the conventional encoding.
                    self.out.write_all(b"n")?;
                    self.out.write_all(&x.to_le_bytes())?;
                } else {
                    return Err(IoError::InvalidNumber {
                        value: x,
                        location: "numeric output".into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Bare integer field (segment headers like `Ci`, counts, etc.). Always
    /// followed by an end-of-line / end-of-record marker, emit it yourself
    /// via `eol_or_comment` or `newline`.
    pub(crate) fn int(&mut self, n: i64) -> io::Result<()> {
        match self.opts.format {
            NlFormat::Ascii => write!(self.out, "{n}"),
            NlFormat::Binary => self.out.write_all(&nl_i32(n, "integer field")?.to_le_bytes()),
        }
    }

    /// Bare double field (bound, RHS).
    pub(crate) fn dbl(&mut self, x: f64) -> Result<(), IoError> {
        match self.opts.format {
            NlFormat::Ascii => {
                let s = self.fmt_value(x)?;
                write!(self.out, "{s}")?;
            }
            NlFormat::Binary => {
                if !x.is_finite() && !self.opts.nonfinite_strings {
                    return Err(IoError::InvalidNumber {
                        value: x,
                        location: "numeric output".into(),
                    });
                }
                self.out.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Segment introducer like `C0` / `O0` / `J3 5` / `S0 8 zork`. The first
    /// argument is the letter; remaining ints are appended space-separated in
    /// ASCII (`C0`, `O0 0`, `J3 5`), or as binary fields in binary mode. An
    /// optional name suffix (e.g. suffix segment label) is appended.
    pub(crate) fn seg_header(
        &mut self,
        letter: u8,
        ints: &[i64],
        name: Option<&str>,
    ) -> io::Result<()> {
        match self.opts.format {
            NlFormat::Ascii => {
                self.out.write_all(&[letter])?;
                for (i, n) in ints.iter().enumerate() {
                    if i == 0 {
                        write!(self.out, "{n}")?;
                    } else {
                        write!(self.out, " {n}")?;
                    }
                }
                if let Some(s) = name {
                    write!(self.out, " {s}")?;
                }
                writeln!(self.out)?;
            }
            NlFormat::Binary => {
                self.out.write_all(&[letter])?;
                for n in ints {
                    self.out.write_all(&nl_i32(*n, "segment field")?.to_le_bytes())?;
                }
                if let Some(s) = name {
                    self.out.write_all(&nl_i32(s.len(), "name length")?.to_le_bytes())?;
                    self.out.write_all(s.as_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// End-of-row marker. ASCII: `\n`. Binary: nothing.
    pub(crate) fn eor(&mut self) -> io::Result<()> {
        if self.opts.format == NlFormat::Ascii {
            writeln!(self.out)?;
        }
        Ok(())
    }

    /// Space separator between ints/doubles inside a record. ASCII only.
    pub(crate) fn sep(&mut self) -> io::Result<()> {
        if self.opts.format == NlFormat::Ascii {
            self.out.write_all(b" ")?;
        }
        Ok(())
    }

    fn fmt_value(&self, x: f64) -> Result<String, IoError> {
        if !x.is_finite() {
            if self.opts.nonfinite_strings {
                return Ok(if x.is_nan() {
                    "NaN".into()
                } else if x > 0.0 {
                    "Infinity".into()
                } else {
                    "-Infinity".into()
                });
            }
            return Err(IoError::InvalidNumber { value: x, location: "numeric output".into() });
        }
        if (x - x.trunc()).abs() == 0.0 && x.abs() < 1e16 {
            #[expect(clippy::cast_possible_truncation)]
            let n = x as i64;
            return Ok(format!("{n}"));
        }
        if let Some(prec) = self.opts.precision {
            let prec = prec.max(1) as usize;
            // `prec` significant digits in scientific notation, e.g. "3.142e0".
            return Ok(format!("{:.*e}", prec.saturating_sub(1), x));
        }
        Ok(format!("{x}"))
    }
}

/// Convert a count/index/opcode to the 4-byte signed int used by binary NL
/// records, failing with an error rather than silently truncating a value that
/// does not fit in `i32`.
fn nl_i32<T>(value: T, field: &str) -> io::Result<i32>
where
    T: TryInto<i32> + Copy + std::fmt::Display,
{
    value.try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("NL binary {field} {value} does not fit in i32"),
        )
    })
}

fn as_short(x: f64) -> Option<i16> {
    if (x - x.trunc()).abs() == 0.0 && (-32768.0..=32767.0).contains(&x) {
        #[expect(clippy::cast_possible_truncation)]
        let v = x as i16;
        Some(v)
    } else {
        None
    }
}

fn as_long(x: f64) -> Option<i32> {
    if (x - x.trunc()).abs() == 0.0 && (-2_147_483_648.0..=2_147_483_647.0).contains(&x) {
        #[expect(clippy::cast_possible_truncation)]
        let v = x as i32;
        Some(v)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::nl_i32;

    #[test]
    fn nl_i32_accepts_in_range() {
        assert_eq!(nl_i32(5_i64, "f").unwrap(), 5);
        assert_eq!(nl_i32(7_u32, "f").unwrap(), 7);
        assert_eq!(nl_i32(i32::MAX as usize, "f").unwrap(), i32::MAX);
    }

    #[test]
    fn nl_i32_rejects_overflow() {
        assert!(nl_i32(i64::from(i32::MAX) + 1, "f").is_err());
        assert!(nl_i32(u32::MAX, "f").is_err());
    }
}
