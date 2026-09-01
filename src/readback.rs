//! Getting tensors off the device, on targets that can block and targets that cannot.
//!
//! Burn computes lazily: a tensor is a handle to work that may not have run
//! yet, and reading it is inherently asynchronous. Every backend therefore
//! offers `into_data_async`, and a synchronous `into_data` that drives it to
//! completion.
//!
//! That second one is not available everywhere. `into_data` polls the future
//! exactly once on `wasm32` and panics if it is still pending, because a
//! browser has no way to block a thread on a GPU buffer map. Natively there is
//! no such restriction. So the same call is a normal read on one target and an
//! unconditional panic on another, decided by the backend rather than by
//! anything visible at the call site.
//!
//! This module is the crate's only answer to that. Every readback goes through
//! [`blocking`], which returns [`SetFitError::Readback`] where `into_data`
//! would have panicked — a failure a caller can match on and recover from, and
//! whose message names the `_async` method that does work there.
//!
//! The consequence for the rest of the crate is that `_async` methods are the
//! real implementations and the synchronous ones are thin wrappers. Writing it
//! the other way around would mean a second implementation of every loop that
//! reads a tensor, kept in step by hand.

use crate::error::{Result, SetFitError};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, try_read_sync};
use core::future::Future;

/// Drive a future to completion on the host thread.
///
/// Natively this blocks and always succeeds. On `wasm32` it polls once, so it
/// succeeds only for a backend whose reads complete immediately — `ndarray`
/// does, a GPU backend does not — and otherwise reports which method to call
/// instead of panicking inside Burn.
pub(crate) fn blocking<T>(what: &str, fut: impl Future<Output = Result<T>>) -> Result<T> {
    match try_read_sync(fut) {
        Some(result) => result,
        None => Err(SetFitError::Readback(format!(
            "{what} needs to wait for the device, and this target cannot block. \
             A browser has no synchronous GPU readback, so on wasm32 a GPU \
             backend can only be driven through the `_async` methods; call \
             those, or use the `ndarray` backend, whose reads complete \
             immediately."
        ))),
    }
}

/// Bring a tensor back to the host as floats.
///
/// The one place a `TensorData` conversion failure is turned into an error, so
/// callers deal in `Vec<f32>` and not in two failure modes.
pub(crate) async fn floats<B: Backend, const D: usize>(t: Tensor<B, D>) -> Result<Vec<f32>> {
    t.into_data_async()
        .await
        .map_err(|e| SetFitError::Readback(format!("reading a tensor from the device: {e}")))?
        .into_vec::<f32>()
        .map_err(|e| SetFitError::Store(format!("{e:?}")))
}

/// Bring a one-element tensor back to the host.
pub(crate) async fn scalar<B: Backend>(t: Tensor<B, 1>) -> Result<f32> {
    Ok(floats(t)
        .await?
        .first()
        .copied()
        // `into_scalar` would panic on an empty tensor. Every caller here builds
        // this from a reduction, so it is unreachable rather than merely unlikely.
        .unwrap_or(f32::NAN))
}
