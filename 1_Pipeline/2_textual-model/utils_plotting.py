import plotly.express as px
import pandas as pd

def plot_value_counts_with_avg(df, column_name, title=None):
    # Count values
    data_counts = df[column_name].value_counts().reset_index()
    data_counts.columns = [column_name, 'Counts']

    # Plot
    fig = px.bar(data_counts, x=column_name, y='Counts', title=title or f'{column_name} Value Counts',
                 text='Counts', color='Counts', color_continuous_scale='Blues')

    # Average line
    avg = data_counts['Counts'].mean()
    fig.add_hline(y=avg, line_dash="dash", line_color="red",
                  annotation_text=f"Average: {avg:.2f}", annotation_position="bottom right")

    # Layout tweaks
    fig.update_layout(
        xaxis_title=column_name,
        yaxis_title='Frequency',
        coloraxis_showscale=False,
        xaxis_tickangle=-45
    )

    fig.show()


import pandas as pd
import plotly.express as px

def analyze_capitalization_patterns(df, column="text"):
    starts_with_cap = 0
    fully_capitalized = 0
    total_words = 0

    for text in df[column].dropna():
        words = text.split()
        total_words += len(words)
        for word in words:
            if word.isupper():
                fully_capitalized += 1
            elif word[0].isupper() and word[1:].islower():
                starts_with_cap += 1

    # Calculate proportions
    starts_with_cap_pct = starts_with_cap / total_words * 100
    fully_capitalized_pct = fully_capitalized / total_words * 100

    # Prepare data for plotly
    data = pd.DataFrame({
        "Capitalization Type": ["Starts with Capital", "Fully Capitalized"],
        "Proportion (%)": [starts_with_cap_pct, fully_capitalized_pct]
    })

    # Add total word count as annotation in the title
    title = f"🧠 Capitalization Patterns (Out of {total_words:,} Words)"

    # Create Plotly bar chart
    fig = px.bar(
        data,
        x="Capitalization Type",
        y="Proportion (%)",
        text=data["Proportion (%)"].map(lambda x: f"{x:.2f}%"),
        color="Capitalization Type",
        title=title
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        yaxis=dict(range=[0, max(data["Proportion (%)"]) * 1.2]),
        template="plotly_white"
    )

    fig.show()


import pandas as pd
import plotly.express as px

def plot_fully_capitalized_words(df, column="text", top_n=30):
    cap_words = {}

    for text in df[column].dropna():
        for word in text.split():
            if word.isupper() and len(word) > 1:  # Ignore things like "I"
                cap_words[word] = cap_words.get(word, 0) + 1

    # Prepare DataFrame
    cap_df = pd.DataFrame(cap_words.items(), columns=["Word", "Count"])
    cap_df = cap_df.sort_values("Count", ascending=False).head(top_n)

    # Plot with Plotly
    fig = px.bar(
        cap_df,
        x="Word",
        y="Count",
        text="Count",
        title=f"🔠 Top {top_n} Fully Capitalized Words",
        labels={"Word": "Fully Capitalized Word", "Count": "Frequency"},
    )
    fig.update_traces(marker_color='crimson', textposition="outside")
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis=dict(showgrid=True),
        template="plotly_white",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    fig.show()



import re

# 1. Define emoji regex once
emoji_regex = re.compile("[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002700-\U000027BF"  # dingbats
    u"\U000024C2-\U0001F251"  # enclosed characters
    "]+", flags=re.UNICODE)

# 2. Check if text contains emojis
def contains_emoji(text):
    return bool(emoji_regex.search(text))

# 3. Extract all emojis from text
def extract_emojis(text):
    return emoji_regex.findall(text)