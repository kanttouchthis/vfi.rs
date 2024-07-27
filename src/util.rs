use tch::{Device, Kind, Tensor};

pub fn get_options() -> (Kind, Device) {
    let device = Device::cuda_if_available();
    let kind = if device.is_cuda() {
        Kind::Half
    } else {
        Kind::Float
    };
    (kind, device)
}

pub fn reverse_channels(t: &mut Tensor) {
    // assumes dim == 4 -> BCHW, dim == 3 -> HWC
    let dim = t.dim();
    let c = match dim {
        4 => t.size()[1], // BCHW
        3 => t.size()[2], // HWC
        _ => panic!(
            "Invalid input shape for reverse_channels: t.dim() needs to be 3 or 4, but found {}",
            dim
        ),
    };

    let indices = Tensor::arange(c, (Kind::Int64, t.device())).flip(0);

    *t = match dim {
        4 => t.index_select(1, &indices),
        3 => t.index_select(2, &indices),
        _ => unreachable!(),
    };
}
