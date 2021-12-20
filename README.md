# SemanticTitleSearch

We have two lists of words.
List #1 - has 10,000 "free style" text entries describing people's job titles. While in general similar jobs have similar texts, they usually never look the same.
E.g.: "customer success manager, support", "customer success associate", "customer success representative I", "customer success representative II", ... - are all "customer success". This is of course a simple example.

List #2 - has a list of 100's of "organized" job titles which we want to use.

The ask is to build the code that for each item in List #1 finds what is the item in List #2 that is the most similar in terms of text and meaning.


## Using the tool

Open the `example.ipynb` notebook. The default parameter is to output the top 3 matches using the `topn` parameter. This can be changed. User should run
the `fit` and `predict` method. The `predict` method outputs a saved `.json` file to `output_matches/top_matches.json`
