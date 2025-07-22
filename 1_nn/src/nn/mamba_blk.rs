use crate::nn::mamba;

use super::{
    Attention, Context, Distribution, Mlp, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::destruct,
};

#[derive(Clone)]
pub struct MambaBlk<T> {
    pub attn_norm: Normalization<T>,
    pub mixer: Attention<T>,
    pub all_reduce: bool,
}

impl<T> MambaBlk<T> {
    #[inline]
    pub const fn new(
        attn_norm: Normalization<T>,
        mixer: Attention<T>,
    ) -> Self {
        Self {
            attn_norm,
            mixer,
            all_reduce: false,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> MambaBlk<TPTensor<T>> {
        let Self {
            attn_norm,
            mixer,
            ..
        } = self;
        MambaBlk{
            attn_norm: attn_norm.tensor_parallel(),
            mixer: mixer.tensor_parallel(dist),
            all_reduce: !dist.is_mono(),
        }
    }
}

impl<T> NuralNetwork<T> for MambaBlk<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            attn_norm,
            mixer,
            all_reduce,
        } = self;

        destruct!([x, pos] = inputs);
        let residual = x.clone();
        let tensors = ctx.trap("attn-norm", attn_norm , [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("mixer", mixer, [x, pos, residual])?;


        let tensors = if all_reduce {
            ctx.call("", "all-reduce", Some("sum".into()), tensors)?
        } else {
            tensors
        };

        Ok((ctx, tensors))

    }
}
