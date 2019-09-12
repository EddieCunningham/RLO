docker build -t mcmc_rlo -f Dockerfile .
docker run -it --rm -v "$(pwd)":/app/host -p 8888:8888 mcmc_rlo