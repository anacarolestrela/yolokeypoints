#!/bin/bash
//só funciona pra linux/wsl


mkdir -p frames

for video in *.mp4; do
  base=$(basename "$video" .mp4)
  mkdir -p frames/"$base"
  
  # Este comando pega um frame a cada 10
  ffmpeg -i "$video" -vf "select=not(mod(n\,10))" -vsync vfr frames/"$base"/frame_%04d.png
done