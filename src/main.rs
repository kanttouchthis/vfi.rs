use tch::Tensor;
mod model;
mod util;

fn main() {
    let (kind, device) = util::get_options();
    println!("Cuda:{}", device.is_cuda());
    let ti = Tensor::randn([1, 16, 360, 640], (kind, device));
    println!("{:?}", ti.size());
    let tf = Tensor::randn([1, 2, 360, 640], (kind, device));
    println!("{:?}", tf.size());
    let r = model::warp(&ti, &tf);
    println!("{:?}", r.size());
}
