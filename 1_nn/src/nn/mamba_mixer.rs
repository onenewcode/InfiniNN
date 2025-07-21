use crate::{nn::mamba, weight_types::RowTPWeight, Linear, TPAction};

use super::{
    Attention, Context, Distribution, Mlp, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::destruct,
};

#[derive(Clone)]
pub struct  MambaMixer<T> {
    pub mixer_conv: Normalization<T>,
    pub mixer_act: Attention<T>,
    pub mixer_in: Linear<T>,
    pub mixer_x: Linear<T>,
    pub mixer_dt: Linear<T>,
        pub mixer_out: Linear<T>,
}

impl<T>  MambaMixer<T> {
    #[inline]
    pub const fn new(
    mixer_conv: Normalization<T>,
    mixer_act: Attention<T>,
    mixer_in: Linear<T>,
    mixer_x: Linear<T>,
    mixer_dt: Linear<T>,
     mixer_out: Linear<T>,
    ) -> Self {
        Self {
            mixer_conv,
            mixer_act,
            mixer_in,
            mixer_x,
            mixer_dt,
            mixer_out,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) ->  MambaMixer<TPTensor<T>> {
        let Self {
            mixer_conv,
            mixer_act,
            mixer_in,
            mixer_x,
            mixer_dt,
            mixer_out,
        } = self;
        todo!("待确定 MambaMixer 的 tensor_parallel 实现");
         MambaMixer{
            mixer_conv: mixer_conv.tensor_parallel(),
            mixer_act: mixer_act.tensor_parallel(dist),
            mixer_in: mixer_in.parallel(TPAction::new(RowTPWeight, dist)),
            mixer_x: mixer_x.parallel(TPAction::new(RowTPWeight, dist)),
            mixer_dt: mixer_dt.parallel(TPAction::new(RowTPWeight, dist)),
            mixer_out: mixer_out.parallel(TPAction::new(RowTPWeight, dist)),
        }
    }
}

impl<T> NuralNetwork<T> for  MambaMixer<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            mixer_conv,
            mixer_act,
            mixer_in,
            mixer_x,
            mixer_dt,
            mixer_out,
        } = self;
        // TODO: Implement the launch logic for MambaBackbone
        todo!("Implement MambaBackbone launch logic");
        destruct!([x, pos] = inputs);
        let residual = x.clone();
        let tensors = ctx.trap("attn-norm", mixer_act, [x])?;
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
