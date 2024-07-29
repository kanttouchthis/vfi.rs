# vfi.rs
Video Frame Interpolation in rust using [interpany-clearer](https://github.com/zzh-tech/InterpAny-Clearer) in [tch](https://crates.io/crates/tch/). Currently only supports the RIFE_sdi_recur architecture. Approximately 2-4x faster than the python implementation.

# Acknowledgements
```bibtex
@article{zhong2023clearer,
  title={Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation},
  author={Zhong, Zhihang and Krishnan, Gurunandan and Sun, Xiao and Qiao, Yu and Ma, Sizhuo and Wang, Jian},
  journal={arXiv preprint arXiv:2311.08007},
  year={2023}
}
```
