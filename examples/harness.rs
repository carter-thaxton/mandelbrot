//! Play the evaluation scenarios (see `mandelbrot::harness`).
//!
//! `cargo run --release --example harness -- [SCENARIO...] [--size WxH] [--csv DIR] [--png DIR] [--master]
//!   Real runs: frame times and image quality per scenario; --png saves each
//!   scenario's final frame and its deep reference.
//!
//! `cargo run --release --example harness -- --compat record|check FILE [SCENARIO...]`
//!   Deterministic runs: record their fingerprints to FILE, or check them
//!   against it (exit status 1 on any difference).

use mandelbrot::harness::{self, Config, Summary};
use mandelbrot::render;

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut names, mut csv, mut compat) = (vec![], None, None);
    let mut png: Option<String> = None;
    let mut master = false;
    let mut cfg = Config::real();
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').ok_or("--size WxH")?;
                cfg.width = w.parse().map_err(|_| "bad width")?;
                cfg.height = h.parse().map_err(|_| "bad height")?;
            }
            "--csv" => csv = Some(val()?),
            "--png" => png = Some(val()?),
            "--master" => master = true,
            "--compat" => compat = Some((val()?, val()?)),
            name => names.push(name.to_string()),
        }
    }
    if compat.is_some() {
        cfg = Config::compat();
    }
    if master {
        cfg.settings = mandelbrot::policy::Settings::master();
    }
    let all = harness::scenarios();
    let known: Vec<_> = all.iter().map(|s| s.name).collect();
    if let Some(bad) = names.iter().find(|n| !known.contains(&n.as_str())) {
        return Err(format!("unknown scenario {bad} (have {})", known.join(", ")));
    }
    let chosen: Vec<_> = all.into_iter().filter(|s| names.is_empty() || names.iter().any(|n| n == s.name)).collect();

    let mut r = render::create_headless()?;
    let mut fingerprints = String::new();
    let mode = match (cfg.deterministic, cfg.settings == mandelbrot::policy::Settings::master()) {
        (true, true) => "deterministic, master",
        (true, false) => "deterministic",
        (false, true) => "real, master",
        (false, false) => "real",
    };
    println!("{mode}, {}x{} at {} fps\n{}", cfg.width, cfg.height, harness::FPS, Summary::HEADER);
    for s in &chosen {
        let start = std::time::Instant::now();
        let run = harness::run(s, &cfg, r.as_mut())?;
        if let Some(dir) = &csv {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            std::fs::write(format!("{dir}/{}.csv", s.name), run.csv()).map_err(|e| e.to_string())?;
        }
        if let Some(dir) = &png {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let images = [("", &run.final_image), ("-deep", &run.final_deep)];
            for (suffix, image) in images {
                if let Some(rgba) = image {
                    write_png(&format!("{dir}/{}{suffix}.png", s.name), cfg.width, cfg.height, rgba)?;
                }
            }
        }
        fingerprints += &run.fingerprint();
        println!("{}  ({:.1} s)", run.summary().row(), start.elapsed().as_secs_f64());
    }
    match compat {
        Some((mode, file)) if mode == "record" => {
            std::fs::write(&file, fingerprints).map_err(|e| e.to_string())?;
            println!("recorded {file}");
        }
        Some((mode, file)) if mode == "check" => {
            let expected = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
            // Only the scenarios that were played.
            let expected: String = expected
                .lines()
                .filter(|l| chosen.iter().any(|s| l.split_whitespace().next() == Some(s.name)))
                .map(|l| format!("{l}\n"))
                .collect();
            harness::compare_fingerprints(&expected, &fingerprints)?;
            println!("identical to {file}");
        }
        Some((mode, _)) => return Err(format!("--compat {mode}: expected record or check")),
        None => {}
    }
    Ok(())
}

fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgba).map_err(|e| e.to_string())
}
