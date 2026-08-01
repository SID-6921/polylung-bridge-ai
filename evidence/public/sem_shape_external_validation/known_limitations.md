# Known limitation: `bead` class data scarcity

The `bead` class is substantially underrepresented relative to `fibre` and `fragment`
throughout this pilot, and this drives its poor and unreliable performance in external
validation. In the original Toronto (Microplastics_SEM) training data, `bead` accounts
for only 36 of 166 training images and 8 of 36 validation images, versus 63-67
training images for `fibre` and `fragment`. In the independent CUNY biosolids
external validation set, `bead` has only 7 examples out of 697 total. All three
architectures evaluated performed poorly on this class under external validation:
Swin-T achieved 2.6% precision / 42.9% recall (F1 = 0.05), EfficientNet-B0 achieved
3.7% precision / 28.6% recall (F1 = 0.07), and ResNet50 achieved 0% precision / 0%
recall (F1 = 0.00) on `bead`, compared to substantially stronger performance on
`fibre` and `fragment` for all models. This pattern -- a class with few training
examples showing near-zero precision out-of-distribution while abundant classes
degrade more gracefully -- reflects a genuine data scarcity problem rather than a
flaw specific to any one architecture. Combined with the earlier finding that the
original in-dataset near-100% accuracy was inflated by a session-level confound
(see `data_leakage_notes.md`), this suggests the `bead` class in particular has
likely been learned from a narrow, session-specific visual signature rather than
generalizable bead morphology. Reliable bead classification would require
substantially more `bead`-class training data drawn from multiple independent
imaging sessions and, ideally, multiple source products/instruments, so that the
model learns morphology that generalizes rather than instrument- or
session-specific artifacts. Until such data is collected, `bead` predictions from
any of these models should be treated as unreliable.
