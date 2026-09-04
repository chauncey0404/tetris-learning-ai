@echo off
title V8.8.2 NVIDIA GPU Monitor
echo ================================================================
echo V8.8.2 NVIDIA GPU MONITOR - 200ms sampling
echo ================================================================
echo Watch utilization.gpu, power.draw and clocks.sm while the profiler runs.
echo Press Ctrl+C to stop.
echo.
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem --format=csv --loop-ms=200
