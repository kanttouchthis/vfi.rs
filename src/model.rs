use super::util;
use tch::{nn, Device, Kind, Tensor};

pub fn warp(ten_input: &Tensor, ten_flow: &Tensor) -> Tensor {
    let (batch_size, height, width) = (ten_flow.size()[0], ten_flow.size()[2], ten_flow.size()[3]);
    let kind = ten_flow.kind();
    let device = ten_flow.device();

    let ten_horizontal = Tensor::linspace(-1.0, 1.0, width, (kind, device))
        .view([1, 1, 1, width])
        .expand([batch_size, -1, height, -1], false);

    let ten_vertical = Tensor::linspace(-1.0, 1.0, height, (kind, device))
        .view([1, 1, height, 1])
        .expand([batch_size, -1, -1, width], false);

    let mut grid = Tensor::cat(&[ten_horizontal, ten_vertical], 1);

    let flow_x = ten_flow.narrow(1, 0, 1) / ((width as f64 - 1.0) / 2.0);
    let flow_y = ten_flow.narrow(1, 1, 1) / ((height as f64 - 1.0) / 2.0);
    grid += Tensor::cat(&[flow_x, flow_y], 1);

    grid = grid.permute([0, 2, 3, 1]);
    ten_input.grid_sampler(&grid, 0, 1, true)
}

fn conv2d_prelu(vs: &nn::Path, i: i64, o: i64, k: i64, stride: i64) -> nn::Sequential {
    let config = nn::ConvConfig {
        stride: stride,
        padding: 1,
        ..Default::default()
    };
    let prelu_weight = (vs / "1").ones("weight", &[o]);
    nn::seq()
        .add(nn::conv2d(vs / "0", i, o, k, config))
        .add_fn(move |xs: &Tensor| xs.prelu(&prelu_weight))
}

fn double_conv2d_prelu(vs: &nn::Path, i: i64, o: i64, k: i64) -> nn::Sequential {
    nn::seq()
        .add(conv2d_prelu(&(vs / "conv1"), i, o, k, 2))
        .add(conv2d_prelu(&(vs / "conv2"), o, o, k, 1))
}

fn deconv2d(vs: &nn::Path, i: i64, o: i64) -> nn::Sequential {
    let config = nn::ConvTransposeConfig {
        stride: 2,
        padding: 1,
        ..Default::default()
    };
    let prelu_weight = (vs / "1").ones("weight", &[o]);

    nn::seq()
        .add(nn::conv_transpose2d(&(vs / "0"), i, o, 4, config))
        .add_fn(move |xs: &Tensor| xs.prelu(&prelu_weight))
}

pub struct IFBlock {
    conv0: nn::Sequential,
    convblock: nn::Sequential,
    lastconv: nn::ConvTranspose2D,
}

impl IFBlock {
    pub fn new(vs: &nn::Path, i: i64, c: i64) -> IFBlock {
        let conv0 = nn::seq()
            .add(conv2d_prelu(&(vs / "conv0" / "0"), i, c / 2, 3, 2))
            .add(conv2d_prelu(&(vs / "conv0" / "1"), c / 2, c, 3, 2));

        let convblock = nn::seq()
            .add(conv2d_prelu(&(vs / "convblock" / "0"), c, c, 3, 1))
            .add(conv2d_prelu(&(vs / "convblock" / "1"), c, c, 3, 1))
            .add(conv2d_prelu(&(vs / "convblock" / "2"), c, c, 3, 1))
            .add(conv2d_prelu(&(vs / "convblock" / "3"), c, c, 3, 1))
            .add(conv2d_prelu(&(vs / "convblock" / "4"), c, c, 3, 1))
            .add(conv2d_prelu(&(vs / "convblock" / "5"), c, c, 3, 1))
            .add(conv2d_prelu(&(vs / "convblock" / "6"), c, c, 3, 1))
            .add(conv2d_prelu(&(vs / "convblock" / "7"), c, c, 3, 1));

        let lastconv_config = nn::ConvTransposeConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let lastconv = nn::conv_transpose2d(&(vs / "lastconv"), c, 5, 4, lastconv_config);

        IFBlock {
            conv0,
            convblock,
            lastconv,
        }
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        flow: Option<&Tensor>,
        scale: Option<f64>,
    ) -> (Tensor, Tensor) {
        let scale: f64 = match scale {
            Some(v) => v,
            None => 1.0f64,
        };

        let mut xs = match scale {
            1.0f64 => xs.shallow_clone(),
            _ => xs.upsample_bilinear2d(
                &[xs.size()[2] / scale as i64, xs.size()[3] / scale as i64],
                false,
                None,
                None,
            ),
        };

        xs = match flow {
            Some(v) => Tensor::cat(
                &[
                    xs,
                    v.upsample_bilinear2d(
                        &[v.size()[2] / scale as i64, v.size()[3] / scale as i64],
                        false,
                        None,
                        None,
                    ) / scale,
                ],
                1,
            ),
            None => xs,
        };

        xs = xs.apply(&self.conv0);
        xs = xs.apply(&self.convblock) + xs;
        xs = xs.apply(&self.lastconv);
        xs = xs.upsample_bilinear2d(
            &[
                xs.size()[2] * (scale as i64) * 2,
                xs.size()[3] * (scale as i64) * 2,
            ],
            false,
            None,
            None,
        );
        (xs.narrow(1, 0, 4) * scale * 2.0, xs.narrow(1, 4, 1))
    }
}

pub struct ContextNet {
    conv0: nn::Sequential,
    conv1: nn::Sequential,
    conv2: nn::Sequential,
    conv3: nn::Sequential,
}

impl ContextNet {
    pub fn new(vs: &nn::Path, c: i64) -> ContextNet {
        let conv0 = double_conv2d_prelu(&(vs / "conv1"), 3, c, 3);
        let conv1 = double_conv2d_prelu(&(vs / "conv2"), c, c * 2, 3);
        let conv2 = double_conv2d_prelu(&(vs / "conv3"), c * 2, c * 4, 3);
        let conv3 = double_conv2d_prelu(&(vs / "conv4"), c * 4, c * 8, 3);
        ContextNet {
            conv0,
            conv1,
            conv2,
            conv3,
        }
    }

    pub fn forward(&self, xs: &Tensor, flow: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        let mut xs = xs.apply(&self.conv0);
        let mut flow = flow.upsample_bilinear2d(
            &[flow.size()[2] / 2 as i64, flow.size()[3] / 2 as i64],
            false,
            None,
            None,
        ) / 2.0;
        let f0 = warp(&xs, &flow);

        xs = xs.apply(&self.conv1);
        flow = flow.upsample_bilinear2d(
            &[flow.size()[2] / 2 as i64, flow.size()[3] / 2 as i64],
            false,
            None,
            None,
        ) / 2.0;
        let f1 = warp(&xs, &flow);

        xs = xs.apply(&self.conv2);
        flow = flow.upsample_bilinear2d(
            &[flow.size()[2] / 2 as i64, flow.size()[3] / 2 as i64],
            false,
            None,
            None,
        ) / 2.0;
        let f2 = warp(&xs, &flow);

        xs = xs.apply(&self.conv3);
        flow = flow.upsample_bilinear2d(
            &[flow.size()[2] / 2 as i64, flow.size()[3] / 2 as i64],
            false,
            None,
            None,
        ) / 2.0;
        let f3 = warp(&xs, &flow);

        (f0, f1, f2, f3)
    }
}

pub struct Unet {
    down0: nn::Sequential,
    down1: nn::Sequential,
    down2: nn::Sequential,
    down3: nn::Sequential,
    up0: nn::Sequential,
    up1: nn::Sequential,
    up2: nn::Sequential,
    up3: nn::Sequential,
    conv: nn::Conv2D,
}

impl Unet {
    pub fn new(vs: &nn::Path, c: i64) -> Unet {
        let down0 = double_conv2d_prelu(&(vs / "down0"), 17, c * 2, 3);
        let down1 = double_conv2d_prelu(&(vs / "down1"), c * 4, c * 4, 3);
        let down2 = double_conv2d_prelu(&(vs / "down2"), c * 8, c * 8, 3);
        let down3 = double_conv2d_prelu(&(vs / "down3"), c * 16, c * 16, 3);

        let up0 = deconv2d(&(vs / "up0"), c * 32, c * 8);
        let up1 = deconv2d(&(vs / "up1"), c * 16, c * 4);
        let up2 = deconv2d(&(vs / "up2"), c * 8, c * 2);
        let up3 = deconv2d(&(vs / "up3"), c * 4, c);

        let config = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv = nn::conv2d(&(vs / "conv"), c, 3, 3, config);

        Unet {
            down0,
            down1,
            down2,
            down3,
            up0,
            up1,
            up2,
            up3,
            conv,
        }
    }

    pub fn forward(
        &self,
        img0: &Tensor,
        img1: &Tensor,
        warped_img0: &Tensor,
        warped_img1: &Tensor,
        mask: &Tensor,
        flow: &Tensor,
        c0: (Tensor, Tensor, Tensor, Tensor),
        c1: (Tensor, Tensor, Tensor, Tensor),
    ) -> Tensor {
        let s0 =
            Tensor::cat(&[img0, img1, warped_img0, warped_img1, mask, flow], 1).apply(&self.down0);
        let s1 = Tensor::cat(&[s0.shallow_clone(), c0.0, c1.0], 1).apply(&self.down1);
        let s2 = Tensor::cat(&[s1.shallow_clone(), c0.1, c1.1], 1).apply(&self.down2);
        let s3 = Tensor::cat(&[s2.shallow_clone(), c0.2, c1.2], 1).apply(&self.down3);
        let mut xs = Tensor::cat(&[s3, c0.3, c1.3], 1).apply(&self.up0);

        xs = Tensor::cat(&[xs, s2], 1).apply(&self.up1);
        xs = Tensor::cat(&[xs, s1], 1).apply(&self.up2);
        xs = Tensor::cat(&[xs, s0], 1).apply(&self.up3);
        xs = xs.apply(&self.conv);

        xs.sigmoid()
    }
}

pub struct IFNetsdi {
    block0: IFBlock,
    block1: IFBlock,
    block2: IFBlock,
    contextnet: ContextNet,
    unet: Unet,
}

impl IFNetsdi {
    pub fn new(vs: &nn::Path) -> IFNetsdi {
        let vs = &(vs / "module");
        let block0 = IFBlock::new(&(vs / "block0"), 11, 240);
        let block1 = IFBlock::new(&(vs / "block1"), 22, 150);
        let block2 = IFBlock::new(&(vs / "block2"), 22, 90);
        let contextnet = ContextNet::new(&(vs / "contextnet"), 16);
        let unet = Unet::new(&(vs / "unet"), 16);

        IFNetsdi {
            block0,
            block1,
            block2,
            contextnet,
            unet,
        }
    }

    pub fn load(path: &str, kind: Kind, device: Device) -> Self {
        let mut vs = nn::VarStore::new(device);
        let model = IFNetsdi::new(&(vs.root()));
        vs.load(path).unwrap();
        vs.set_kind(kind);
        model
    }

    pub fn forward(
        &self,
        img0: &Tensor,
        img1: &Tensor,
        sdi_map: &Tensor,
        scales: &[f64],
    ) -> Tensor {
        // Iteration 0
        let (mut flow, mut mask) = self.block0.forward(
            &Tensor::cat(&[img0, img1, sdi_map], 1),
            None,
            Some(scales[0]),
        );

        let mut warped_img0 = warp(img0, &flow.narrow(1, 0, 2));
        let mut warped_img1 = warp(img1, &flow.narrow(1, 2, 2));

        // Iteration 1
        let (_flow, _mask) = self.block1.forward(
            &Tensor::cat(&[img0, img1, sdi_map, &warped_img0, &warped_img1, &mask], 1),
            Some(&flow),
            Some(scales[1]),
        );

        flow += _flow;
        mask += _mask;

        warped_img0 = warp(img0, &flow.narrow(1, 0, 2));
        warped_img1 = warp(img1, &flow.narrow(1, 2, 2));

        // Iteration 2
        let (_flow, _mask) = self.block2.forward(
            &Tensor::cat(&[img0, img1, sdi_map, &warped_img0, &warped_img1, &mask], 1),
            Some(&flow),
            Some(scales[2]),
        );

        flow += _flow;
        mask += _mask;

        warped_img0 = warp(img0, &flow.narrow(1, 0, 2));
        warped_img1 = warp(img1, &flow.narrow(1, 2, 2));

        let mut merged: Tensor = &warped_img0 * &mask + &warped_img1 * (1.0 - &mask);

        let c0 = self.contextnet.forward(img0, &flow.narrow(1, 0, 2));
        let c1 = self.contextnet.forward(img1, &flow.narrow(1, 2, 2));

        let mut res =
            self.unet
                .forward(img0, img1, &warped_img0, &warped_img1, &mask, &flow, c0, c1);

        res = res.narrow(1, 0, 3) * 2.0 - 1.0;

        merged = merged + res;

        merged.clamp(0.0, 1.0)
    }

    fn _inference(
        &self,
        img0: &Tensor,
        img1: &Tensor,
        sdi_map: &Tensor,
        iters: i64,
        scales: &[f64],
    ) -> Tensor {
        let iters = match iters {
            0 => 1,
            _ => iters,
        };

        let mut img_cur = img0.shallow_clone();
        let mut _sdi_map = sdi_map.shallow_clone();

        for i in 0..iters {
            let v1 = ((i as f64) + 1.0f64) / (iters as f64);
            let v2 = (i as f64) / (iters as f64);
            _sdi_map = Tensor::cat(&[sdi_map * v1, sdi_map * v2, img_cur], 1);
            img_cur = self.forward(img0, img1, &_sdi_map, scales);
        }

        img_cur
    }

    pub fn inference(
        &self,
        img0: &Tensor,
        img1: &Tensor,
        num: i64,
        iters: i64,
        scales: Option<&[f64]>,
        kind: tch::Kind,
        device: tch::Device,
    ) -> Vec<Tensor> {
        let _guard = tch::no_grad_guard();
        let img0 = util::preprocess(img0, kind, device, true);
        let img1 = util::preprocess(img1, kind, device, true);

        let scale = match scales {
            Some(scale) => scale,
            None => &[4.0, 2.0, 1.0],
        };

        let (bs, h, w) = (img0.size()[0], img0.size()[2], img0.size()[3]);
        let (kind, device) = (img0.kind(), img1.device());

        let mut sdi_maps: Vec<Tensor> = Vec::new();

        for i in 1..num + 1 {
            let v = (i as f64) / ((num as f64) + 1.0f64);
            sdi_maps.push(Tensor::full(&[bs, 1, h, w], v, (kind, device)));
        }

        let mut results: Vec<Tensor> = Vec::new();
        for sdi_map in sdi_maps {
            results.push(self._inference(&img0, &img1, &sdi_map, iters, scale));
        }

        results
    }
}
