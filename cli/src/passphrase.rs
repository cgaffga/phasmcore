// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use std::io::IsTerminal;

/// Get a passphrase from -p flag or interactive prompt.
/// `confirm` = true for encode (type twice), false for decode (single entry).
pub fn get_passphrase(
    flag_value: Option<&str>,
    prompt_label: &str,
    confirm: bool,
) -> Result<String, CliError> {
    if let Some(p) = flag_value {
        return Ok(p.to_string());
    }

    if !std::io::stdin().is_terminal() {
        return Err(CliError::InvalidArgs(
            "Passphrase required in non-interactive mode. Use -p <passphrase>.".into(),
        ));
    }

    let passphrase = rpassword::prompt_password(format!("{prompt_label}: "))
        .map_err(|e| CliError::Io(e))?;

    if passphrase.is_empty() {
        return Err(CliError::InvalidArgs("Passphrase cannot be empty.".into()));
    }

    if confirm {
        let again = rpassword::prompt_password(format!("{prompt_label} (confirm): "))
            .map_err(|e| CliError::Io(e))?;
        if passphrase != again {
            return Err(CliError::PassphraseMismatch);
        }
    }

    Ok(passphrase)
}
