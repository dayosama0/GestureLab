# iPhone, Windows laptop and next model iteration

Updated 2026-11-07. Current working inference release remains unchanged.

## Transfer to another Windows computer

Run `.venv/Scripts/python scripts/package_transfer.py` on the current PC. The local archive is exports/gesturelab-laptop.zip; it contains source code, trained checkpoints, experiment metrics/plots and report. It omits .venv, node_modules, Git internals, the original source dataset and demo video. It contains a private image-bearing report and is for transfer to your own laptop, not public distribution. Add --include-data only to include the supplied original ZIP and source documents for your own use.

Transfer this archive by your chosen Wi-Fi file-sharing method or personal storage; no USB cable is required. No external upload is performed by the packaging script. On the laptop install Python 3.10+ and Node.js, extract the archive, open the gesturelab folder in VS Code, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

Open http://127.0.0.1:5173. Do not retrain: trained weights are included. The environment is rebuilt on the laptop, so internet is needed for dependency installation. TRANSFER_MANIFEST.json records SHA256 per included file. Portability is prepared and archive hashes tested here; actual laptop operation remains unverified until run there.

## iPhone without a cable

The quickest app-like route is a web app opened in Safari and added to the Home Screen via Share / Add to Home Screen. Model inference can remain on the PC or laptop. This does not provide autonomous offline iPhone inference. Both devices can connect over the same Wi-Fi.

Current development server binds to127.0.0.1, so it is not yet reachable from the phone. A phone-facing HTTPS endpoint and install metadata must be configured. Plain HTTP to a LAN IP is not a secure context for getUserMedia; the localhost exemption refers to the phone itself, not the remote PC. Local trusted HTTPS needs certificate trust setup on the iPhone; a public HTTPS tunnel is an alternative but would expose traffic to an external provider and is not enabled. No firewall weakening or certificate installation has been performed.

Apple guide: https://support.apple.com/en-ie/guide/iphone/iphea86e5236/27/ios/27
Camera secure context: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

## Observed generalization issue and new data

The user reports poor live thumbs-up and peace recognition. The saved MobileNetV4 source-test matrix contains11/15 correct peace and15/15 thumbs_up; those small familiar-split results do not establish live-camera robustness. Crop, viewpoint, person/background differences, data coverage, normalization adaptation and frozen-backbone learning are hypotheses to investigate rather than proven causes.

The owner states the original source was the HaGRID GitHub repository: https://github.com/hukenovs/hagrid. This is reported provenance, not yet a verified mapping from the400 supplied filenames to upstream records. The official repository includes annotations, hand boxes and user_id; investigate an additional balanced subset of fist, palm, peace and like (mapping to fist, open_palm, peace, thumbs_up). Keep source val/test unchanged, deduplicate all additional training sources against them, and use participant/session IDs to prevent leakage. Create a separate new-person live-camera holdout. No new images have been downloaded or trained as of this note. Preserve licenses and source attribution.

Confirmed GPU via nvidia-smi: NVIDIA GeForce GTX1660Ti,6144MiB VRAM, driver560.94. Existing PyTorch environment is CPU-only and existing transfer backbones are frozen. Next training can use a separately prepared CUDA environment and staged fine-tuning with a measured feasible batch size; do not replace verified checkpoints or retune against the original test.

## Assignment interpretation

Assignment2.docx includes the explicit alternative: Or you can choose a new/modern SOTA created so far. Therefore Step-Video-T2V/ArtAug are the suggested gesture-specific generative route, not the sole allowed method. Current implementation follows the alternative classifier route; it does not implement either generator.

Official Step-Video documentation lists72-79GB peak GPU memory for its documented video settings, recommends80GB GPUs, and has CUDA capability constraints in the text encoder. The documented configuration is not suitable for the current6GB GTX1660Ti: https://github.com/stepfun-ai/Step-Video-T2V

ArtAug is a synthesis-understanding interaction method that improves text-to-image generation, not a name for ordinary brightness/contrast augmentation: https://arxiv.org/abs/2412.12888
