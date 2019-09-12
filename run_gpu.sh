docker build -t mcmc_rlo_gpu -f DockerfileJaxGPU .
docker run --gpus all -it -v "$(pwd)":/app/host -p 8888:8888 mcmc_rlo_gpu