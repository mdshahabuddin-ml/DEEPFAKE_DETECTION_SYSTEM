from src.dataset import get_multidataset_splits

train_samples, val_samples, test_samples = get_multidataset_splits(
    indian_root="data/indian",
    wild_root="data/wilddeepfake",
    dfdc_root="data/dfdc",
    dfdc_metadata="data/dfdc/metadata.json",
    ff_root="data/ff",
    celebdf_root="data/celebdf",

    max_frames_per_video=20,
    seed=42,

    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
)


def count(samples):
    real = sum(label == 0 for _, label in samples)
    fake = sum(label == 1 for _, label in samples)
    return real, fake


print("=" * 60)
print("TRAINING DISTRIBUTION")
print("=" * 60)

for name, samples in [
    ("TRAIN", train_samples),
    ("VALIDATION", val_samples),
    ("TEST", test_samples),
]:
    real, fake = count(samples)

    print(f"\n{name}")
    print(f"Total: {len(samples)}")
    print(f"REAL : {real}")
    print(f"FAKE : {fake}")

print("\n" + "=" * 60)
