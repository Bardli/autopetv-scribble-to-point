# syntax=docker/dockerfile:1
#
# Rebuild recipe for the FightTumor AutoPET V final candidate. The two DKFZ
# checkpoints must be placed under third_party/autoPET-interactive/_model
# before building; they are intentionally ignored by Git.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3

LABEL org.autopetv.base_image="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3"
LABEL org.autopetv.build_id="2fold_tta_20260830b"
LABEL org.autopetv.build_variant="TARGET_DEFAULT_CU126"
LABEL org.autopetv.delivery_lock="dkfz_2fold_tta_equal_logits"

RUN groupadd -r algorithm \
    && useradd -m --no-log-init -r -g algorithm algorithm \
    && mkdir -p /opt/algorithm /input /output/images/tumor-lesion-segmentation \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm
WORKDIR /opt/algorithm

ENV PATH=/home/algorithm/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/algorithm \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    nnUNet_raw=/opt/algorithm/nnUNet_raw \
    nnUNet_preprocessed=/opt/algorithm/nnUNet_preprocessed \
    nnUNet_results=/opt/algorithm/nnUNet_results

COPY --chown=algorithm:algorithm requirements.txt process_entry.py /opt/algorithm/
COPY --chown=algorithm:algorithm baseline /opt/algorithm/baseline
COPY --chown=algorithm:algorithm scripts /opt/algorithm/scripts
COPY --chown=algorithm:algorithm third_party/autoPET-interactive /opt/algorithm/vendor/autoPET-interactive

RUN python scripts/verify_model.py \
        --model-dir /opt/algorithm/vendor/autoPET-interactive/_model \
    && python -m pip install --user -U pip \
    && python -m pip install --user -r requirements.txt \
    && python -m pip install --user --no-deps -e /opt/algorithm/vendor/autoPET-interactive \
    && mkdir -p /opt/algorithm/work/input \
        /opt/algorithm/nnUNet_raw \
        /opt/algorithm/nnUNet_preprocessed \
        /opt/algorithm/nnUNet_results

ENTRYPOINT ["python", "/opt/algorithm/process_entry.py"]
CMD ["--prompt-encoding", "point_edt", "--max-fg-points", "0", "--max-bg-points", "0", "--folds", "0,5"]
