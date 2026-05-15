import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

cgpa         = np.round(np.random.normal(7.2, 0.9, n).clip(4.0, 10.0), 2)
internships  = np.random.choice([0, 1, 2, 3], n, p=[0.35, 0.40, 0.18, 0.07])
projects     = np.random.choice([0, 1, 2, 3, 4], n, p=[0.10, 0.25, 0.35, 0.20, 0.10])
skills       = np.random.randint(2, 10, n)          # number of tech skills
backlogs     = np.random.choice([0, 1, 2, 3], n, p=[0.60, 0.22, 0.12, 0.06])
communication= np.random.choice([1,2,3,4,5], n, p=[0.05,0.15,0.30,0.30,0.20])
aptitude     = np.round(np.random.normal(65, 15, n).clip(20, 100), 1)
branch       = np.random.choice(['CS','IT','ECE','Mech','Civil'], n, p=[0.30,0.20,0.25,0.15,0.10])

# Realistic placement logic
score = (
    (cgpa - 4) / 6 * 35
    + internships * 8
    + projects * 4
    + (skills - 2) / 8 * 10
    - backlogs * 7
    + (communication - 1) / 4 * 10
    + (aptitude - 20) / 80 * 12
    + np.where(branch == 'CS', 5, np.where(branch == 'IT', 3, 0))
    + np.random.normal(0, 5, n)
)
placed = (score > 42).astype(int)

df = pd.DataFrame({
    'cgpa': cgpa,
    'internships': internships,
    'projects': projects,
    'skills': skills,
    'backlogs': backlogs,
    'communication': communication,
    'aptitude_score': aptitude,
    'branch': branch,
    'placed': placed
})

df.to_csv('data/placement_data.csv', index=False)
print(f"Dataset created: {len(df)} students, {df['placed'].sum()} placed ({df['placed'].mean()*100:.1f}%)")
