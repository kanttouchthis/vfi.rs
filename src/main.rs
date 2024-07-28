use tch::{Device, Kind};
mod model;
mod util;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: String,
    #[arg(short, long)]
    output: String,
    #[arg(short, long)]
    model_path: String,
    #[arg(short, long, default_value = "true")]
    keep_audio: bool,
    #[arg(short, long, default_value = "auto")]
    device: String,
    #[arg(long, default_value = "auto")]
    dtype: String,
    #[arg(long, default_value = "2")]
    factor: i64,
    #[arg(long, default_value = "2")]
    iters: i64,
    #[arg(long, default_value = "hevc_nvenc")]
    vcodec: String,
    #[arg(long, default_value = "4M")]
    bitrate: String,
}

fn main() {
    let args = Args::parse();
    let device = match args.device.as_ref() {
        "auto" => Device::cuda_if_available(),
        "cuda" => Device::Cuda(0),
        "cpu" => Device::Cpu,
        _ => Device::Cuda(args.device.parse::<usize>().unwrap()),
    };
    let kind = match args.dtype.as_ref() {
        "float" | "float32" | "fp32" => Kind::Float,
        "half" | "float16" | "fp16" => Kind::Half,
        "bfloat16" | "bf16" => Kind::BFloat16,
        _ => {
            if device.is_cuda() {
                Kind::Half
            } else {
                Kind::Float
            }
        }
    };
    let model = model::IFNetsdi::load(&args.model_path, kind, device);
    println!("Using {:?} {:?}", device, kind);
    util::inference_video(
        &model,
        args.input.as_ref(),
        args.output.as_ref(),
        args.factor,
        args.iters,
        kind,
        device,
        &args.vcodec,
        &args.bitrate,
        args.keep_audio,
    )
}
