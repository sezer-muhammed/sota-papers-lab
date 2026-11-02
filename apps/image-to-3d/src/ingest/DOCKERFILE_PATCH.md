# Dockerfile change needed by Stage A0 (ingest)

`image_to_3d` video ingest runs `ffmpeg` and `ffprobe` as child processes.
The runtime stage of `deploy/docker/Dockerfile` (`nvidia/cuda:*-runtime`) has
neither. This is a note, not an applied edit: the deploy owner applies it.

```diff
 FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
+RUN apt-get update \
+ && apt-get install -y --no-install-recommends ffmpeg \
+ && rm -rf /var/lib/apt/lists/*
 COPY --from=rust /usr/local/bin/spl /usr/local/bin/spl
```

Notes:
- Ubuntu's `ffmpeg` adds roughly 250-300 MB (codec libraries). It decodes
  H.264, HEVC (iPhone), VP9 and AV1 on the CPU, which is what ingest uses by
  default.
- NVDEC (`IngestConfig::ffmpeg_input_args = ["-hwaccel", "cuda"]`) also needs
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` so the container gets
  `libnvcuvid`; Ubuntu's ffmpeg is built with `--enable-cuvid/nvdec`.
- Photos need nothing: JPEG, PNG and WebP decode in Rust (`image` crate).
