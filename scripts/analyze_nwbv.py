import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data (replace with your path)
path = "./../data/OASIS-1/DemographicAndClinicalData/oasis_cross-sectional-5708aa0a98d82080.xlsx"
df = pd.read_excel(path)

# 2. Clean and create AD / CN labels
# Remove patients without a CDR score
df = df.dropna(subset=['CDR'])
# CDR > 0 -> AD, CDR == 0 -> CN
df['Label'] = df['CDR'].apply(lambda x: 'AD' if x > 0 else 'CN')

# 3. Create a figure with two plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Distribution (density curves) ---
sns.histplot(data=df, x='nWBV', hue='Label', kde=True,
             palette={'CN': 'blue', 'AD': 'red'}, ax=axes[0],
             stat='density', common_norm=False, alpha=0.5)
axes[0].set_title('nWBV distribution: AD vs CN')
axes[0].set_xlabel('Normalized Whole Brain Volume (nWBV)')
axes[0].set_ylabel('Density')

# --- Plot 2: Boxplots ---
sns.boxplot(data=df, x='Label', y='nWBV',
            palette={'CN': 'blue', 'AD': 'red'}, ax=axes[1])
axes[1].set_title('Comparison of nWBV medians')
axes[1].set_xlabel('Diagnosis')
axes[1].set_ylabel('nWBV')

plt.tight_layout()
plt.savefig('nwbv_distribution.png')
print("Plot saved as 'nwbv_distribution.png'")
