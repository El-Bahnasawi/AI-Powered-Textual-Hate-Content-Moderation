import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px

class CASE_STUDY:
    def __init__(self):
        # Load dataset
        self.df = pd.read_csv("data/manual_eval/Hate_Speech_Evaluation_Dataset.csv", sep=";")
        self.df_unique_categories = sorted(self.df['category'].dropna().unique())
        self.pivot_df = pd.DataFrame(columns=self.df_unique_categories + ["hate_avg"])
        self.threshold = 0.35
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.compare_models = [
            "Hate-speech-CNERG/dehatebert-mono-english",
            "ctoraman/hate-speech-bert",
            "facebook/roberta-hate-speech-dynabench-r4-target",
            "cardiffnlp/twitter-roberta-base-offensive",
            "cardiffnlp/twitter-roberta-base-hate-latest",
            "medoxz543/hate-speech"
        ]

    def describe_df(self):
        print(f"Number of hate categories: {self.df['category'].nunique()}")
        print(f"Hate categories: {self.df_unique_categories}")
        print(self.df["category"].value_counts().sort_values(ascending=False))

    def modeling(self, model_name, texts):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Tokenization
        encoded_inputs = tokenizer(
            list(texts), 
            return_tensors="pt", 
            padding="max_length",
            max_length=120, 
            truncation=True
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = model(**encoded_inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            hate_probs = probs[:, 1].tolist() if probs.size(1) > 1 else probs[:, 0].tolist()
            predicted_labels = [int(p > self.threshold) for p in hate_probs]


        return predicted_labels

    def populate_df(self):
        for model in self.compare_models:
            print(f"Evaluating model: {model}")
            category_scores = {}

            for category in self.df_unique_categories:
                sub_df = self.df[self.df["category"] == category]
                texts = sub_df["text"].tolist()
                true_labels = sub_df["label"].tolist()
                preds = self.modeling(model, texts)
                accuracy = sum([int(p == t) for p, t in zip(preds, true_labels)]) / len(true_labels)
                category_scores[category] = round(accuracy, 2)

            # Compute average accuracy across hate categories only
            hate_categories = [cat for cat in self.df_unique_categories if cat != "not_hateful"]
            hate_avg = sum([category_scores[cat] for cat in hate_categories]) / len(hate_categories)
            category_scores["hate_avg"] = round(hate_avg, 2)

            # Add results to pivot_df
            self.pivot_df.loc[model] = category_scores

    def show_nonhate_and_hate(self):
        if self.pivot_df.empty:
            print("Run `populate_df()` first to fill in model scores.")
            return

        # Create a new DataFrame for plotting
        plot_df = self.pivot_df[["not_hateful", "hate_avg"]].reset_index()
        plot_df = plot_df.rename(columns={"index": "model"})

        # Melt to long format for Plotly
        melted_df = plot_df.melt(id_vars="model", 
                                value_vars=["not_hateful", "hate_avg"], 
                                var_name="Category", 
                                value_name="Accuracy")

        # Plot
        fig = px.bar(
            melted_df,
            x="model",
            y="Accuracy",
            color="Category",
            barmode="group",
            title="Model Accuracy: Not Hateful vs. Hate Avg",
            text="Accuracy",
            labels={"model": "Model", "Accuracy": "Accuracy", "Category": "Category"}
        )
        fig.update_layout(xaxis_tickangle=-30)
        fig.show()

    def show_heatmap(self):
        if self.pivot_df.empty:
            print("Run `populate_df()` first.")
            return

        # Exclude non-hate categories
        heatmap_df = self.pivot_df.drop(columns=["not_hateful", "hate_avg"], errors="ignore").copy()
        heatmap_df["model"] = heatmap_df.index

        # Melt into long format for Plotly
        melted = heatmap_df.melt(id_vars="model", var_name="Category", value_name="Accuracy")

        # Create heatmap
        fig = px.imshow(
            heatmap_df.drop(columns=["model"]).values,
            labels=dict(x="Hate Category", y="Model", color="Accuracy"),
            x=heatmap_df.drop(columns=["model"]).columns,
            y=heatmap_df["model"],
            color_continuous_scale="RdBu",
            zmin=0,
            zmax=1,
            text_auto=True,
            title="Per-Category Accuracy Heatmap (Hate Categories Only)"
        )

        fig.update_layout(
            xaxis_title="Hate Category",
            yaxis_title="Model",
            xaxis_tickangle=-30
        )

        fig.show()