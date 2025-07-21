use crate::nn::mamba;

use super::{
    Attention, Context, Distribution, Mlp, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::destruct,
};

#[derive(Clone)]
pub struct MambaBackbone<T> {
    pub mixer_before_norm: Normalization<T>,
    pub mixer: Attention<T>,
    pub mixer_after_norm: Normalization<T>,
    pub all_reduce: bool,
}

impl<T> MambaBackbone<T> {
    #[inline]
    pub const fn new(
        mixer_before_norm: Normalization<T>,
        mixer: Attention<T>,
        mixer_after_norm: Normalization<T>,
    ) -> Self {
        Self {
            mixer_before_norm,
            mixer,
            mixer_after_norm,
            all_reduce: false,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> MambaBackbone<TPTensor<T>> {
        let Self {
            mixer_before_norm,
            mixer,
            mixer_after_norm,
            ..
        } = self;
        MambaBackbone{
            mixer_before_norm: mixer_before_norm.tensor_parallel(),
            mixer: mixer.tensor_parallel(dist),
            mixer_after_norm: mixer_after_norm.tensor_parallel(),
            all_reduce: !dist.is_mono(),
        }
    }
}

impl<T> NuralNetwork<T> for MambaBackbone<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            mixer_before_norm,
            mixer,
            mixer_after_norm,
            all_reduce,
        } = self;
        // TODO: Implement the launch logic for MambaBackbone
        todo!("Implement MambaBackbone launch logic");
        destruct!([x, pos] = inputs);
        let residual = x.clone();
        let tensors = ctx.trap("attn-norm", mixer, [x])?;
        destruct!([x] = tensors);
        // let tensors = ctx.trap("attn", attn, [x, pos, residual])?;
        // let tensors = if all_reduce {
        //     ctx.call("", "all-reduce", Some("sum".into()), tensors)?
        // } else {
        //     tensors
        // };

        // destruct!([x] = tensors);
        // let residual = x.clone();
        // let tensors = ctx.trap("ffn-norm", ffn_norm, [x])?;
        // destruct!([x] = tensors);
        // let tensors = ctx.trap("ffn", ffn, [x, residual])?;
        // let tensors = if all_reduce {
        //     ctx.call("", "all-reduce", Some("sum".into()), tensors)?
        // } else {
        //     tensors
        // };

        Ok((ctx, tensors))

    }
}
