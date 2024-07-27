use tch::{nn, Tensor};

pub fn warp(ten_input: &Tensor, ten_flow: &Tensor) -> Tensor {
    let (batch_size, _, height, width) = (
        ten_flow.size()[0],
        ten_flow.size()[1],
        ten_flow.size()[2],
        ten_flow.size()[3],
    );
    let kind = ten_flow.kind();
    let device = ten_flow.device();

    let ten_horizontal = Tensor::linspace(-1.0, 1.0, width, (kind, device))
        .view([1, 1, 1, width])
        .expand([batch_size, -1, height, -1], false);

    let ten_vertical = Tensor::linspace(-1.0, 1.0, height, (kind, device))
        .view([1, 1, height, 1])
        .expand([batch_size, -1, -1, width], false);

    let mut grid = Tensor::cat(&[ten_horizontal, ten_vertical], 1);

    let flow_x = ten_flow.slice(1, 0, 1, 1) / ((width as f64 - 1.0) / 2.0);
    let flow_y = ten_flow.slice(1, 1, 2, 1) / ((height as f64 - 1.0) / 2.0);
    grid += Tensor::cat(&[flow_x, flow_y], 1);

    grid = grid.permute([0, 2, 3, 1]);
    ten_input.grid_sampler(&grid, 0, 1, true)
}

#[derive(Debug)]
struct IFBlock {
    conv0: nn::Sequential,
    convblock: nn::Sequential,
    lastconv: nn::ConvTranspose2D,
}

impl IFBlock {
    fn new(vs: &nn::Path, i: i64, c: i64) -> IFBlock {
        let conv0_config = nn::ConvConfig {
            stride: 2,
            padding: 1,
            dilation: 1,
            ..Default::default()
        };

        let convblock_config = nn::ConvConfig {
            stride: 1,
            padding: 1,
            dilation: 1,
            ..Default::default()
        };

        let lastconv_config = nn::ConvTransposeConfig {
            stride: 1,
            padding: 1,
            dilation: 1,
            ..Default::default()
        };

        let prelu0_0_weight = vs.ones("prelu0_0_weight", &[c / 2]);
        let prelu0_1_weight = vs.ones("prelu0_1_weight", &[c / 2]);

        let prelu1_0_weight = vs.ones("prelu1_0_weight", &[c]);
        let prelu1_1_weight = vs.ones("prelu1_1_weight", &[c]);
        let prelu1_2_weight = vs.ones("prelu1_2_weight", &[c]);
        let prelu1_3_weight = vs.ones("prelu1_3_weight", &[c]);
        let prelu1_4_weight = vs.ones("prelu1_4_weight", &[c]);
        let prelu1_5_weight = vs.ones("prelu1_5_weight", &[c]);
        let prelu1_6_weight = vs.ones("prelu1_6_weight", &[c]);
        let prelu1_7_weight = vs.ones("prelu1_7_weight", &[c]);

        let conv0 = nn::seq()
            .add(nn::conv2d(vs, i, c / 2, 3, conv0_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu0_0_weight))
            .add(nn::conv2d(vs, c / 2, c, 3, conv0_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu0_1_weight));

        let convblock = nn::seq()
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_0_weight))
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_1_weight))
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_2_weight))
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_3_weight))
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_4_weight))
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_5_weight))
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_6_weight))
            .add(nn::conv2d(vs, c, c, 3, convblock_config))
            .add_fn(move |xs: &Tensor| xs.prelu(&prelu1_7_weight));

        let lastconv = nn::conv_transpose2d(vs, c, 5, 4, lastconv_config);

        IFBlock {
            conv0,
            convblock,
            lastconv,
        }
    }
    fn prepare_args_flow(&self, xs: &Tensor, flow: &Tensor, scale: f64) -> Tensor {
        let xs = xs.upsample_bilinear2d(&[xs.size()[2] / scale as i64, xs.size()[3] / scale as i64], false, None, None);
        let flow = flow.upsample_bilinear2d(&[flow.size()[2] / scale as i64, flow.size()[3] / scale as i64], false, None, None) / scale;
        Tensor::cat(&[xs, flow], 1)
    }
    fn prepare_args(&self, xs: &Tensor, scale: f64) -> Tensor {
        xs.upsample_bilinear2d(&[xs.size()[2] / scale as i64, xs.size()[3] / scale as i64], false, None, None)
    }
    fn result_to_flow_mask(&self, xs: &Tensor) -> (Tensor, Tensor){
        (xs.narrow(1, 0, 4)*2.0, xs.narrow(1, 4, 1))
    }
}

impl nn::Module for IFBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.conv0);
        xs = xs.apply(&self.convblock) + xs;
        xs = xs.apply(&self.lastconv);
        xs = xs.upsample_bilinear2d(&[xs.size()[2] * 2, xs.size()[3] * 2], false, None, None);
        xs
    }
}
