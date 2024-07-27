use tch::{nn, Tensor};
use tch::nn::{conv2d, conv_transpose2d, Module, Path, ConvConfig, ConvTransposeConfig};

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

fn if_block_conv0(vs: &Path, i:i64, c:i64){
    let mut config = ConvConfig::default();
    config.stride = 2;
    config.padding = 1;
    config.dilation = 1;
    nn::seq()
        .add(conv2d(vs, i, c / 2, 3, config))
        .add(conv2d(vs, c/2, c, 3, config));
}

fn if_block_conv1(vs: &Path, c:i64){
    let mut config = ConvConfig::default();
    config.stride = 1;
    config.padding = 1;
    config.dilation = 1;
    nn::seq()
        .add(conv2d(vs, c, c, 3, config))
        .add(conv2d(vs, c, c, 3, config))
        .add(conv2d(vs, c, c, 3, config))
        .add(conv2d(vs, c, c, 3, config))
        .add(conv2d(vs, c, c, 3, config))
        .add(conv2d(vs, c, c, 3, config))
        .add(conv2d(vs, c, c, 3, config))
        .add(conv2d(vs, c, c, 3, config));
}

fn if_block_conv2(vs: &Path,  c:i64){
    let mut config = ConvTransposeConfig::default();
    config.stride = 2;
    config.padding = 1;
    //nn::seq()
    //    .add(conv_transpose2d(vs, c, 5, 4, config))
    //    .add_fn(|x| x.prelu());
}
