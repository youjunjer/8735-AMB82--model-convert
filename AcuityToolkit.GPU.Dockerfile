FROM ghcr.io/ameba-aiot/acuity-toolkit:6.18.8

RUN python3 -m pip install --no-cache-dir \
    nvidia-cuda-runtime-cu11 \
    nvidia-cublas-cu11 \
    nvidia-cudnn-cu11==8.6.0.163 \
    nvidia-cufft-cu11 \
    nvidia-curand-cu11 \
    nvidia-cusolver-cu11 \
    nvidia-cusparse-cu11

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.8/site-packages/nvidia/curand/lib:\
/usr/local/lib/python3.8/site-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.8/site-packages/nvidia/cusolver/lib:\
/usr/local/lib/python3.8/site-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.8/site-packages/nvidia/cufft/lib:${LD_LIBRARY_PATH}"
