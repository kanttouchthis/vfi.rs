use super::model;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::str;
use tch::{Device, Kind, Tensor};

pub fn preprocess(img: &Tensor, kind: Kind, device: Device, flip_rgb: bool) -> Tensor {
    let mut img = img.to(device);
    img = match img.dim() {
        3 => img.unsqueeze(0),
        4 => img.copy(),
        _ => panic!(),
    };

    let img_kind = img.kind();
    img = match img_kind {
        Kind::Uint8 => img.to_kind(kind) / 255.0,
        Kind::Float => img.to_kind(kind),
        Kind::Half => img.to_kind(kind),
        Kind::Double => img.to_kind(kind),
        _ => panic!("Can't handle image kind: {:?}", img_kind),
    };

    if flip_rgb {
        reverse_channels(&mut img);
    };
    img
}

pub fn postprocess(img: &Tensor, flip_rgb: bool) -> Tensor {
    let mut img = (img * 255.0).to_kind(Kind::Uint8);
    if flip_rgb {
        reverse_channels(&mut img);
    };
    img.squeeze()
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

fn get_video_info(video_path: &str) -> Result<(f64, u32, u32, u64), Box<dyn std::error::Error>> {
    let output = Command::new("ffprobe")
        .args(&[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,width,height,nb_frames",
            "-of",
            "json",
            video_path,
        ])
        .output()?;

    let stdout = str::from_utf8(&output.stdout)?;
    let json: Value = serde_json::from_str(stdout)?;

    let streams = &json["streams"][0];

    let r_frame_rate = streams["r_frame_rate"]
        .as_str()
        .ok_or("No frame rate found")?;
    let width = streams["width"].as_u64().ok_or("No width found")? as u32;
    let height = streams["height"].as_u64().ok_or("No height found")? as u32;
    let nb_frames = streams["nb_frames"]
        .as_str()
        .ok_or("No nb_frames found")?
        .parse()?;

    // r_frame_rate is in the form "numerator/denominator"
    let parts: Vec<&str> = r_frame_rate.split('/').collect();
    let fps = if parts.len() == 2 {
        let numerator: f64 = parts[0].parse()?;
        let denominator: f64 = parts[1].parse()?;
        numerator / denominator
    } else {
        return Err("Invalid frame rate format".into());
    };

    Ok((fps, width, height, nb_frames))
}

fn buffer_to_tensor(
    buffer: &Vec<u8>,
    width: u32,
    height: u32,
    kind: Kind,
    device: Device,
) -> Tensor {
    let mut img = Tensor::from_slice(buffer);
    img = img.to(device);
    img = img.reshape(&[height as i64, width as i64, 3]);
    img = img.permute(&[2, 0, 1]).unsqueeze(0);
    img = (img / 255.0).to_kind(kind);
    reverse_channels(&mut img);

    img
}

pub fn inference_video(
    model: &model::IFNetsdi,
    input: &str,
    output: &str,
    factor: i64,
    iters: i64,
    kind: Kind,
    device: Device,
    vcodec: &str,
    bitrate: &str,
    keep_audio: bool,
) {
    let num = factor - 1;
    let (fps, width, height, nb_frames) = get_video_info(input).unwrap();
    let temp = output.to_owned() + ".temp";
    let scale = format!("{}x{}", width, height);
    let framesize = (width * height * 3) as usize;
    let audio = input.to_owned() + ".aac";
    let nb_frames_out = nb_frames * factor as u64;

    if keep_audio {
        let mut extract_audio = Command::new("ffmpeg")
            .arg("-i")
            .arg(input)
            .arg("-acodec")
            .arg("copy")
            .arg("-loglevel")
            .arg("error")
            .arg("-hide_banner")
            .arg("-y")
            .arg(&audio)
            .spawn()
            .expect("Failed to open ffmpeg audio process");
        extract_audio
            .wait()
            .expect("Failed to wait for audio reader");
        let audio_path = Path::new(&audio);
        if !audio_path.exists() {
            println!("Audio file not found. Maybe ffmpeg failed?");
            return;
        };
    }

    let mut ffmpeg_read = Command::new("ffmpeg")
        .arg("-hwaccel")
        .arg("cuda")
        .arg("-i")
        .arg(input)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-loglevel")
        .arg("error")
        .arg("-hide_banner")
        .arg("pipe:1")
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to open ffmpeg read process");

    let stdout = ffmpeg_read.stdout.as_mut().expect("Failed to open stdout");
    let mut reader = BufReader::new(stdout);

    let mut ffmpeg_write = Command::new("ffmpeg");
    ffmpeg_write
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-s")
        .arg(scale)
        .arg("-r")
        .arg(format!("{}", fps * factor as f64))
        .arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg("-hide_banner")
        .arg("-i")
        .arg("pipe:0");
    if keep_audio {
        ffmpeg_write.arg("-i").arg(&audio);
    };
    ffmpeg_write
        .arg("-f")
        .arg("mp4")
        .arg("-vcodec")
        .arg(vcodec)
        .arg("-b:v")
        .arg(bitrate)
        .arg(&temp)
        .stdin(Stdio::piped());

    let mut ffmpeg_write = ffmpeg_write
        .spawn()
        .expect("Failed to open ffmpeg write process");
    let stdin = ffmpeg_write.stdin.as_mut().expect("Failed to open stdin");
    let mut writer = BufWriter::new(stdin);

    let mut buffer = vec![0u8; framesize];
    let mut output_buffer;

    reader
        .read_exact(&mut buffer)
        .expect("Failed to read from ffmpeg");

    let bar = ProgressBar::new(nb_frames_out as u64);
    bar.set_style(ProgressStyle::with_template("{bar:40} {per_sec}").unwrap());
    let mut last = buffer_to_tensor(&buffer, width, height, kind, device);
    let mut next: Tensor;
    let mut results: Vec<Tensor>;

    writer
        .write_all(&buffer)
        .expect("Failed to write buffer to ffmpeg");
    bar.inc(1);

    while let Ok(_) = reader.read_exact(&mut buffer) {
        next = buffer_to_tensor(&buffer, width, height, kind, device);

        results = model.inference(&last, &next, num, iters, None, kind, device);

        for mut result in results {
            result = postprocess(&result, false).permute([1, 2, 0]);
            output_buffer = Vec::<u8>::try_from(result.flatten(0, -1))
                .expect("Failed to convert tensor to vec");
            writer
                .write_all(&output_buffer)
                .expect("Failed to write output_buffer to ffmpeg");
            bar.inc(1);
        }
        writer
            .write_all(&buffer)
            .expect("Failed to write buffer to ffmpeg");
        bar.inc(1);
        last = next;
    }

    writer.flush().expect("Failed to flush writer");
    drop(writer);
    ffmpeg_read.wait().expect("Failed to wait for reader");
    ffmpeg_write.wait().expect("Failed to wait for writer");

    let temp_path = Path::new(&temp);
    if !temp_path.exists() {
        println!("Temp file not found. Maybe ffmpeg failed?");
        return;
    };

    fs::rename(temp_path, output).expect(&format!("Failed to rename {} to {}", &temp, &output));
    fs::remove_file(audio).expect("Failed to delete temp audio file");
}
