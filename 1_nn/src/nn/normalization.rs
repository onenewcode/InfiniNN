﻿use super::{Context, NNError, NuralNetwork, Tensor, macros::destruct};
use crate::Dim;
use digit_layout::DigitLayout;

pub struct Normalization<T> {
    pub d: Dim,
    pub epsilon: f64,
    pub items: Type<T>,
}

pub enum Type<T> {
    RmsNorm {
        dt: DigitLayout,
        scale: T,
    },
    LayerNorm {
        dt_scale: DigitLayout,
        scale: T,
        dt_bias: DigitLayout,
        bias: T,
    },
}

impl<T> NuralNetwork<T> for Normalization<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x] = inputs);

        let Self { d, epsilon, items } = self;
        let outputs = match items {
            Type::RmsNorm { dt, scale } => {
                let scale = ctx.load_external("scale", dt, [d], scale);
                ctx.call("", "rms-norm", Some(epsilon.into()), [x, scale])
            }
            Type::LayerNorm {
                dt_scale,
                scale,
                dt_bias,
                bias,
            } => {
                let scale = ctx.load_external("scale", dt_scale, [d.clone()], scale);
                let bias = ctx.load_external("bias", dt_bias, [d], bias);
                ctx.call("", "layer-norm", Some(epsilon.into()), [x, scale, bias])
            }
        };

        Ok((ctx, outputs?))
    }
}
