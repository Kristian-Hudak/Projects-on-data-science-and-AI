# Olympiad Task 1: Recipe Calorie Prediction & Nutritional Optimization

**Competition:** Slovak National Olympiad in Artificial Intelligence (Olympiáda v umelej inteligencii — OAI)  
**Author:** Kristian Hudák  
**Round:** National Qualifier / Home Round

---

## Project Overview

This project tackles automated nutritional analysis and dietary optimization through two complementary tasks:
1. **Calorie Prediction from Natural Language Recipes:** Translating unformatted natural language text (e.g. *"150g pork loin, 200g sweet potatoes, 50g sour cream"*) into precise total caloric predictions. Benchmarks a deep multi-layer neural network against linear regression.
2. **Healthier Recipe Optimization:** Transforming existing recipes into healthier equivalents (reducing caloric density and saturated fats while preserving essential protein content) using both rule-based heuristics and a deep neural network with multi-objective penalty loss.

---

## Directory Structure

```
olympiad_task_1_nutritional_optimization/
├── 01_calorie_calculator/
│   ├── calorie_prediction_neural_network.ipynb     # Calorie prediction: MLP vs Linear Regression
│   └── recipes_dataset.xlsx                        # Curated recipe reference dataset
├── 02_healthier_recipe_optimization/
│   ├── rule_based_recipe_optimization.ipynb        # Heuristic expert system with grammage penalties
│   ├── deep_learning_recipe_optimization.ipynb     # Neural network multi-attribute recipe optimizer
│   ├── method_comparison_heuristics_vs_ai.md       # In-depth technical comparison report
│   └── healthier_ingredients_dataset.csv           # Paired unhealthy-healthy ingredients dataset
└── 03_data_preprocessing/
    ├── usda_nutrition_data_preprocessing.ipynb     # Full ETL pipeline on USDA Foundation Food DB
    ├── df_fdcid_name_calories.xlsx                 # Extracted calorie density lookup table
    └── pivoted_categorized_food_data.xlsx          # Pivoted macronutrient profiles (8 features)
```

---

## Technical Highlights

- **USDA FoodData Central ETL:** Automated ingestion, filtering, and pivoting of Foundation Foods tables, engineering macro ratios ($C/P$, $C/Carb$, $C/Fat$) and categorical mappings.
- **Fuzzy Entity Resolution (`thefuzz`):** Connects informal colloquial ingredient names entered by users to standardized USDA database food descriptions with token set scoring.
- **Why Deep Learning Outperforms Linear Models:** Demonstrates that computing total energy ($\sum \text{Weight}_i \times \text{Density}_i$) is an intrinsically non-linear multiplicative problem that single-layer linear models cannot resolve without non-linear hidden representations.
- **Multi-Objective Loss Formulation:** Incorporates asymmetric penalties to enforce dietary targets:
  $$\mathcal{L}_{\text{penalty}} = 0.5 \cdot \text{kcal} + 1.2 \cdot \text{carbs} + 0.1 \cdot \text{fat} - 1.2 \cdot \text{protein}$$
