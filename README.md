# Real-Time Cough Detection ML Pipeline

A real-time machine learning pipeline that listens to your microphone and detects coughs, printing timestamps when they occur. Designed to work on Mac (both Intel and Apple Silicon), Windows, and Linux.

## Quick Start

### One Command Setup (Recommended)

This downloads datasets, trains the model, and sets everything up automatically.

**macOS / Linux:**
```bash
cd cough_detector
chmod +x run_all.sh
./run_all.sh
```

**Windows:**
```cmd
cd cough_detector
run_all.bat
```

### Run Live Detection

After setup completes:

**macOS / Linux:**
```bash
source venv/bin/activate
python run_detection.py --model checkpoints/best_model.pt --threshold 0.7 --smoothing 1
```

**Windows:**
```cmd
venv\Scripts\activate
python run_detection.py --model checkpoints\best_model.pt --threshold 0.7 --smoothing 1
```

### Manual Setup (Alternative)

If you prefer to run steps individually:

**macOS / Linux:**
```bash
cd cough_detector
chmod +x setup.sh
./setup.sh
source venv/bin/activate
pip install soundfile
python download_esc50.py
python setup_coughvid.py
python train_with_data.py
```

**Windows:**
```cmd
cd cough_detector
setup_windows.bat
venv\Scripts\activate
pip install soundfile
python download_esc50.py
python setup_coughvid.py
python train_with_data.py
```

### 2. Install ffmpeg (required for COUGHVID dataset)

COUGHVID uses `.webm` audio files which require ffmpeg to decode.

**Windows:**
1. Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract the zip to `C:\ffmpeg`
3. You'll have a folder like `C:\ffmpeg\ffmpeg-8.0.1-essentials_build\`
4. Add the `bin` folder to your PATH:
   - Press Windows key, type "environment variables"
   - Click "Edit the system environment variables"
   - Click "Environment Variables..."
   - Under "User variables", select "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin` (adjust for your version)
   - Click OK on all windows
5. **Close and reopen** command prompt
6. Verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 3. Train a Model

**Option A: Quick training (small dataset, ~5 min, weaker results):**
```bash
python train_quick.py
```

**Option B: Full training with COUGHVID (recommended, ~30-60 min, better results):**
```bash
pip install soundfile
python train_quick.py          # Downloads ESC-50 first
python setup_coughvid.py       # Downloads COUGHVID (~1.5GB)
python train_with_data.py      # Trains on combined dataset
```

### 3. Run Live Detection

```bash
# Start listening for coughs
python run_detection.py --model ./checkpoints/best_model.pt
```

When you cough, you'll see output like:
```
🔊 COUGH DETECTED at 2024-01-28 14:32:15.234
   Confidence: 87.3%
```

## Architecture and Model

### Data Pipeline

| Source | Coughs | Non-Coughs | Notes |
|--------|--------|------------|-------|
| COUGHVID | ~3,000 | ~1,500 | Real crowdsourced coughs, high-confidence only (>0.9) |
| ESC-50 | 40 | ~680 | Extended hard negatives (breathing, sneezing, clapping, laughing, keyboard, etc.) |
| **Total** | **~3,040** | **~2,180** | All real recordings, no synthetic data |

### Preprocessing Pipeline

```
Raw Audio File (.wav, .webm, .ogg)
       ↓
  Resample to 16kHz mono
       ↓
  Normalize amplitude to [-1, 1]
       ↓
  Pad or trim to exactly 1 second (16,000 samples)
       ↓
  Extract Features:
  ┌─────────────────────────────────────┐
  │ Mel Spectrogram (64 bands)          │
  │   • FFT size: 512                   │
  │   • Window: 25ms (400 samples)      │
  │   • Hop: 10ms (160 samples)         │
  │   • Frequency: 100Hz - 4000Hz       │  ← Bandpass focused on cough frequencies
  │   • Convert to dB, normalize [0,1]  │
  ├─────────────────────────────────────┤
  │ MFCCs (13 coefficients)             │  ← Captures vocal tract characteristics
  │   • Normalized to zero mean/std     │
  ├─────────────────────────────────────┤
  │ MFCC Deltas (13 coefficients)       │  ← Rate of change over time
  │   • First derivative of MFCCs       │
  └─────────────────────────────────────┘
       ↓
  Stack vertically: 64 + 13 + 13 = 90 features
       ↓
  Output: 2D tensor (1, 90, 101)
          (1 channel, 90 features, 101 time frames)
```

**Why these features?**
- **Mel Spectrogram (64 bands):** Mimics human ear perception, captures frequency content
- **MFCCs (13):** Standard for speech/audio - captures vocal tract characteristics of coughs
- **MFCC Deltas (13):** Captures dynamics - coughs have distinctive rapid onset
- **100-4000Hz bandpass:** Coughs primarily contain energy in this range, filters out irrelevant frequencies

### Model Architecture (CoughDetectorResidual)

```
Input: (batch, 1, 90, 101)  ← 90 features (mel + MFCC + delta) × 101 time frames
              ↓
┌─────────────────────────────────────┐
│ Conv2D(1 → 32, 7×7, stride=2)       │
│ BatchNorm2D(32)                     │
│ ReLU                                │
│ MaxPool2D(2×2)                      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Residual Block (32 → 64)            │
│   ├─ Conv2D(3×3) + BN + ReLU        │
│   ├─ Conv2D(3×3) + BN               │
│   └─ Skip connection (1×1 conv)     │  ← Helps gradient flow
│ ReLU                                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Residual Block (64 → 128)           │
│   ├─ Conv2D(3×3) + BN + ReLU        │
│   ├─ Conv2D(3×3) + BN               │
│   └─ Skip connection (1×1 conv)     │
│ ReLU                                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ AdaptiveAvgPool2D(1×1)              │
│ Dropout(0.5)                        │
│ Linear(128 → 2)                     │
└─────────────────────────────────────┘
              ↓
Output: (batch, 2)  ← logits for [non_cough, cough]
```

**Parameters:** ~200,000

**Why Residual Architecture?**
- **Skip connections**: Prevents vanishing gradients, allows deeper networks
- **Better feature learning**: Can learn both fine and coarse patterns
- **Proven architecture**: Based on ResNet, state-of-the-art for image classification
- **Moderate size**: Good balance between capacity and speed

### Training Configuration

| Setting | Value |
|---------|-------|
| Loss | CrossEntropyLoss with dynamic class weights |
| Optimizer | AdamW (lr=0.0005, weight_decay=0.01) |
| Scheduler | Cosine annealing with warm restarts |
| Batch size | 32 |
| Max epochs | 150 |
| Early stopping | Patience of 20 epochs |
| Augmentation | Time shift, volume, gaussian noise, SpecAugment |

### Real-Time Inference Pipeline

```
Microphone Input (continuous)
         ↓
    100ms audio chunks
         ↓
┌────────────────────────────┐
│ Sliding window buffer      │
│ (1 sec window, 250ms hop)  │
└────────────────────────────┘
         ↓
    Mel spectrogram
         ↓
    CNN prediction → probability
         ↓
┌────────────────────────────┐
│ Smoothing (avg last 3)     │  ← Reduces noise
│ Threshold check (0.7)      │
│ Debounce (0.5 sec)         │  ← Prevents double-triggers
└────────────────────────────┘
         ↓
   🔊 COUGH DETECTED + timestamp
```

## Key Technical Choices

### 1. Combined Feature Set (Mel + MFCC + Delta)
**Choice:** 90 features combining mel spectrograms, MFCCs, and MFCC deltas.

**Why:**
- **Mel spectrograms (64):** Capture frequency content in a perceptually-relevant scale
- **MFCCs (13):** Standard in speech/audio recognition, capture spectral envelope
- **MFCC deltas (13):** Capture temporal dynamics - critical for cough's sharp onset
- Combined features give the model multiple "views" of the same audio

### 2. Bandpass Filtering (100-4000 Hz)
**Choice:** Restrict frequency range to 100-4000 Hz.

**Why:**
- Coughs primarily contain energy in this range
- Filters out low-frequency rumble (HVAC, traffic)
- Filters out high-frequency noise (hiss, electronics)
- Focuses model capacity on relevant frequencies

### 3. Residual Architecture
**Choice:** ResNet-style architecture with skip connections.

**Why:**
- **Skip connections:** Prevent vanishing gradients, enable deeper networks
- **~200K parameters:** Good balance between capacity and inference speed
- **Proven architecture:** Based on ResNet, works well for audio classification
- Outperformed simpler CNN and depthwise-separable variants in our testing

### 3. Data Augmentation Strategy
**Choice:** Both waveform and spectrogram augmentation.

**Waveform augmentations:**
- Time shifting (±20%)
- Speed perturbation (0.9x - 1.1x)
- Volume perturbation (0.7x - 1.3x)
- Gaussian noise (SNR 10-30dB)
- Background noise mixing (if noise samples available)

**Spectrogram augmentations:**
- SpecAugment: frequency masking (up to 10 bands)
- SpecAugment: time masking (up to 20 frames)

**Why:**
- Cough sounds vary significantly between people
- Need robustness to recording conditions (noise, volume, microphone quality)
- Limited labeled cough data necessitates aggressive augmentation

### 4. Sliding Window with Debouncing
**Choice:** 1-second analysis windows with 250ms hop and 0.5s debounce.

**Why:**
- 1 second captures full cough duration (typical cough: 200-500ms)
- 250ms hop provides responsive detection
- Debouncing prevents multiple triggers for a single cough event

### 5. Confidence Smoothing
**Choice:** Average predictions over 3 consecutive windows.

**Why:**
- Single-frame predictions can be noisy
- Smoothing reduces false positives without significantly increasing latency
- Coughs are sustained events, not instantaneous

## Data Sources

### Primary: COUGHVID Dataset (Recommended)
- ~25,000 crowdsourced cough recordings
- We filter to ~3,000 high-confidence samples (confidence > 0.8)
- Source: https://zenodo.org/records/4048312
- **License:** Creative Commons Attribution 4.0

### Secondary: ESC-50 Dataset
- 2000 environmental audio recordings across 50 classes
- 40 cough samples (class 24)
- ~680 hard negative samples from 17 similar classes
- Source: https://github.com/karolpiczak/ESC-50
- **License:** Creative Commons Attribution Non-Commercial

**Hard negative classes from ESC-50:**
- Human sounds: breathing, snoring, sneezing, laughing, crying baby
- Transient sounds: clapping, door knock, mouse click, keyboard typing
- Other: dog bark, can opening, clock alarm, washing machine, vacuum cleaner

### Data Balance
| Source | Coughs | Non-Coughs | 
|--------|--------|------------|
| COUGHVID | ~3,000 | ~1,500 |
| ESC-50 | 40 | ~680 |
| **Total** | **~3,040** | **~2,180** |

## Training

### Quick Training (ESC-50 only)

```bash
python train_quick.py
```

### Full Training with Custom Data

```bash
# Prepare data (downloads ESC-50 and organizes it)
python prepare_data.py --output-dir ./data

# Train with custom data
python src/train.py \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --model-type small \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | None | Directory with cough/non_cough subdirs |
| `--output-dir` | ./checkpoints | Where to save models |
| `--model-type` | small | Model architecture (small/standard/residual) |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--device` | auto | Compute device (auto/cpu/cuda/mps) |
| `--no-esc50` | False | Disable ESC-50 dataset |

### Expected Training Results

With COUGHVID + ESC-50 (recommended):
- Validation Accuracy: ~87%
- Validation F1 (cough class): **~0.87**
- Precision: ~78% (78% of detections are real coughs)
- Recall: ~92% (detects 92% of actual coughs)
- Training time: ~30-60 minutes on GPU, ~2-3 hours on CPU

With ESC-50 alone (quick training):
- Validation F1: ~0.15-0.20 (not enough cough samples)
- Use only for testing setup, not production

## Running Live Detection

### Basic Usage

```bash
python run_detection.py --model ./checkpoints/best_model.pt
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Path to trained model |
| `--threshold` | 0.7 | Detection confidence threshold (0-1) |
| `--smoothing` | 3 | Predictions to average |
| `--debounce` | 0.5 | Minimum seconds between detections |
| `--device` | auto | Compute device |
| `--audio-device` | None | Audio input device index |
| `--list-devices` | - | List available audio devices |

### Adjusting Sensitivity

**More sensitive (more detections, possibly more false positives):**
```bash
python run_detection.py --model ./checkpoints/best_model.pt --threshold 0.5 --smoothing 2
```

**Less sensitive (fewer false positives, might miss quiet coughs):**
```bash
python run_detection.py --model ./checkpoints/best_model.pt --threshold 0.85 --smoothing 4
```

### Selecting Audio Input

```bash
# List available devices
python run_detection.py --list-devices

# Use specific device
python run_detection.py --model ./checkpoints/best_model.pt --audio-device 2
```

## Project Structure

```
cough_detector/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── preprocessing.py     # Audio preprocessing and mel spectrograms
│   ├── model.py            # Neural network architectures
│   ├── augmentation.py     # Data augmentation techniques
│   ├── dataset.py          # Dataset classes and data loading
│   ├── train.py            # Training loop and utilities
│   └── inference.py        # Real-time inference engine
├── checkpoints/            # Saved model checkpoints
├── datasets/               # Downloaded datasets (ESC-50)
├── data/                   # Prepared training data
├── run_detection.py        # Main entry point for live detection
├── train_quick.py          # Quick training script
├── prepare_data.py         # Data preparation utilities
├── setup.sh                # Environment setup script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Troubleshooting

### "No audio backend available"
```bash
# macOS
brew install portaudio
pip install sounddevice

# Linux
sudo apt-get install portaudio19-dev
pip install sounddevice
```

### "Permission denied for microphone"
On macOS, grant microphone permission to Terminal/your IDE in System Preferences → Security & Privacy → Microphone.

### "CUDA out of memory"
Use CPU for inference (it's fast enough):
```bash
python run_detection.py --model ./checkpoints/best_model.pt --device cpu
```

### High false positive rate
1. Increase threshold: `--threshold 0.8`
2. Increase smoothing: `--smoothing 5`
3. Train with more diverse negative samples

### Missing cough detections
1. Decrease threshold: `--threshold 0.5`
2. Decrease debounce: `--debounce 0.3`
3. Check microphone levels - speak/cough into microphone

## License

This project uses the ESC-50 dataset which is licensed under Creative Commons Attribution Non-Commercial. The code is provided as-is for educational and research purposes.

## Acknowledgments

- ESC-50 Dataset: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification"
- PyTorch and torchaudio teams
- Sounddevice library developers
