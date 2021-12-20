import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import pickle
from collections import defaultdict
import json


class SemanticJobMatcher:
    """
    Semantic Job Matcher is created with a fit and predict method.

    Parameter
    ---------
    topn: int, default value is 3. user can change to return the top N matches

    Methods
    -------
    Fit: method takes two input lists. The method embeds each lists and creates
    a permutation of each combination

    Perdict: method calculates the cosine similarity between each embedded title
    and returns the top N matches based on cosine.

    """

    def __init__(self, topn=3):
        self.model = SentenceTransformer(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )
        self.topn = topn

    def fit(self, list1, list2):
        self.embedded_titles = [
            self.model.encode(title, show_progress_bar=True) for title in [list1, list2]
        ]
        self.permutations = pd.DataFrame(
            list(itertools.product(list1, list2)), columns=["SearchTitle", "JobTitle"]
        )

    def predict(self):
        self.permutations["cosine_scores"] = cosine_similarity(
            self.embedded_titles[0], self.embedded_titles[1]
        ).reshape(-1, 1)
        self.permutations["ranking"] = self.permutations.groupby("SearchTitle")[
            "cosine_scores"
        ].rank(method="dense", ascending=False)

        filtered = self.permutations[
            self.permutations["ranking"] <= self.topn
        ].sort_values(["SearchTitle", "ranking"])
        titles_dict = filtered.pivot_table(
            "JobTitle", index="SearchTitle", columns="ranking", aggfunc="max"
        ).to_dict("index")
        scores_dict = filtered.pivot_table(
            "cosine_scores", index="SearchTitle", columns="ranking", aggfunc="max"
        ).to_dict("index")

        output = defaultdict(list)

        for d in (titles_dict, scores_dict):
            for key, value in d.items():
                output[key].append(value)

        with open("output_matches/top_matches.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        return output
