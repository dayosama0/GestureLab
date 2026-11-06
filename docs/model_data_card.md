# Data and model card - expanded edition

The 400 original UUIDs and labels match official HaGRID annotations and 400 distinct upstream user IDs, with zero original split intersections. This verifies separation under upstream identifiers, not independent identity or personal consent. The source photographs are external HaGRID data, not claimed as personally collected images.

The expansion adds 2800 real ROIs: 2000 train, 400 validation and 400 test. All 400 original users are excluded from additions. Adding the original 280 training images yields 2280 training records. The original 60-image test and the fresh 400-image test (292 users) are reported separately. Selection validation contains 400 images from218 users. Logical peace adds 25% official peace_inverted examples while retaining the four project labels and upstream source gestures.

Sources use matching official hand boxes with15% margin, RGB conversion and gray128 letterbox to320. Source/crop hashes, ZIP CRC checks, annotations and user joins are saved. HaGRID has a custom public license derived from Attribution-ShareAlike4.0 but explicitly not a Creative Commons license. Actual terms are cached at data/hagrid_cache/HaGRID_license_en_us.pdf. Keep image-bearing evidence private until applicable sharing rights are confirmed.

Legacy transfer models use frozen features and224/ImageNet preprocessing. New ResNet18 uses224/ImageNet; new MobileNetV4 uses native256 and0.5 mean/std. Both new runs optimized the entire backbone after head warmup. Metadata records actual epochs, validation selection, CUDA precision, weight changes and checkpoint hashes. ResNet uses AMP; Mobile uses verified stable GPU float32. See real run JSONs for measured results.

The browser supports square/tall/wide manual ROI and displays the exact prepared image alongside genuine Grad-CAM. No automatic hand localization is claimed. Upload/camera requests are memory-only and not logged or retained. Softmax is uncalibrated; attribution is approximate; this four-way classifier cannot reliably reject unrelated objects.

The expanded comparison changes several factors together and does not isolate their individual causal effects. The original noaug/moderate pair remains the conventional augmentation ablation, with its fivefold exposure confound disclosed. Personal phone images remain reserved for a future final check.

References: official dataset https://github.com/hukenovs/hagrid ; MobileNetV4 https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php ; model https://huggingface.co/timm/mobilenetv4_conv_small.e3600_r256_in1k ; Grad-CAM https://arxiv.org/abs/1610.02391 . No ArtAug or Step-Video implementation is claimed. The specific Neurohive citation choice remains for instructor confirmation.
