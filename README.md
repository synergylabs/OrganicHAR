# OrganicHAR: Towards Activity Discovery in Organic Settings for Privacy Preserving Sensors Using Efficient Video Analysis

[[paper (IMWUT 2025)](https://doi.org/10.1145/3770674)]


**Abstract:**
Deploying human activity recognition (HAR) at home is still rare because sensor signals vary wildly across houses, people,
and time, essentially requiring in-situ data collection and training. Prior approaches use cameras to generate training labels
for privacy-preserving sensors (LiDAR, RADAR, Thermal), but this forces sensors to detect predefined activities that cameras
can see yet the sensors themselves cannot reliably distinguish. In this work, we introduce OrganicHAR, an activity discovery
framework that inverts this relationship by placing sensor capabilities at the center of activity discovery. Our approach
identifies naturally occurring signal patterns using privacy-preserving sensors, leverages Vision Language Models (VLMs) only
during these key moments for scene understanding, and discovers discrete activity labels at granularities that these sensors
can reliably detect. Our evaluation with 12 participants demonstrates OrganicHAR’s effectiveness: it achieves 79% accuracy
for coarse (4-5) activities using only basic ambient sensors (radar, lidar, thermal arrays), and 73% accuracy for fine-grained
(8-9) activities when a wearable IMU, depth, and pose sensor are added. OrganicHAR maintains 77% accuracy on average
across configurations while discovering 4-8 categories per user (15 across all users) tailored to each environment and sensor
capabilities. By triggering video processing only at key moments identified by local sensors, we reduce queries to VLM by
90%, enabling practical and privacy-preserving activity recognition in natural settings.


## Reference

```bibtex
@INPROCEEDINGS{patidar25organichar,
    title = {OrganicHAR: Towards Activity Discovery in Organic Settings for Privacy Preserving Sensors Using Efficient Video Analysis},
    author = {Prasoon Patidar, Riku Arakawa, Ricardo Graça, Rúben Moutinho, Adriano Soares, Ana Vasconcelos, Filippo Talami, Joana
Couto da Silva, Inês Silva, Cristina Mendes Santos, Mayank Goel, and Yuvraj Agarwal},
    journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies}
    year = {2025},
    publisher = {ACM},
    address = {Shanghai, China},
    article = {203},
    volume = {9},
    number = {4},
    month = {12},
    doi = {https://doi.org/10.1145/3770674},
    numpages = {32},
    keywords = {ubiquitous sensing, privacy first design, human activity recognition},
}

