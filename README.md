# Applied Machine Learning Project — Which Wisdom Book Is This Passage From?

## Project Description

A complete supervised machine learning workflow in Python (NumPy, Pandas, scikit-learn, Matplotlib and Seaborn), built in a single Jupyter notebook, `modeling.ipynb`.
It classifies passages from eight wisdom books — four biblical (Proverbs, Ecclesiastes, Ecclesiasticus, Book of Wisdom) and four Asian (Upanishads, Yoga Sutras, Buddhist Sutras, Tao Te Ching) — by their source book, using the word counts of each passage.
The notebook compares a majority-class baseline, Multinomial Naive Bayes and a regularized logistic regression on TF-IDF features with repeated stratified cross-validation, evaluates the selected model on a held-out test set, measures how stable the test score is across random splits, and checks whether the model relies on the archaic English of the biblical translations.

**Dataset:** *A Study of Asian Religious and Biblical Texts* — UCI Machine Learning Repository (2019), CC BY 4.0 — https://doi.org/10.24432/C55S4W
File used (copy in `data/`): `AllBooks_baseline_DTM_Labelled.csv`, 590 passages × 8,267 columns (a passage label plus 8,266 word counts).

## How to Run the Project

Requirements: **Python 3.12 or newer** and Git. The pinned versions in `requirements.txt` (for example `numpy==2.5.3`, `scipy==1.18.1` and `scikit-learn==1.9.1`) do not install on Python 3.11 or older.

```bash
git clone https://github.com/arturoappp/applied-machine-learning-project.git
cd applied-machine-learning-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook modeling.ipynb
```

Then run all cells (Kernel → Restart & Run All). The notebook runs top to bottom without errors and writes four figures to `figures/`. A full run takes about 10 minutes on a laptop CPU, because model selection uses 5 × 5 repeated cross-validation and the stability check retrains the models on 20 splits.

To regenerate the dependency file from the project environment:

```bash
pip freeze > requirements.txt
```

## Repository Structure

```
modeling.ipynb                          # inspection, preprocessing, model selection, evaluation, bias check, summary
Machine_Learning_Analysis_Report.pdf    # written report with in-text citations and references
requirements.txt                        # exact package versions (pip freeze)
data/                                   # AllBooks_baseline_DTM_Labelled.csv
figures/                                # fig1_model_comparison, fig2_confusion_matrix, fig3_top_words, fig4_split_stability
```
