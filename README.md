# GA-Text-Summarization

[Extractive text summarization using genetic algorithms](https://arxiv.org/abs/2105.02365)

If you found our code useful for research, please use the following BibTeX entry for citation.
```BibTeX
@misc{chen2021genetic,
      title={Genetic Algorithms For Extractive Summarization}, 
      author={William Chen and Kensal Ramos and Kalyan Naidu Mullaguri},
      year={2021},
      eprint={2105.02365},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Install Dependencies

```
pip install requirements.txt
```

## Format Dataset

This splits the corpus in the stories folder into a body (the actual article) and highlights (the summary). Does not split dataset into training and testing.
```
cd src
python dataset.py
```

## Split into Train and Test

The program assumes that the dataset is split into training and testing in the following manner. There is no script included for automatic splitting.
```
GA-Text-Summarization\src\dataset\train\body\sample.txt
GA-Text-Summarization\src\dataset\train\highlights\sample.txt

GA-Text-Summarization\src\dataset\test\body\sample.txt
GA-Text-Summarization\src\dataset\test\highlights\sample.txt
```

## Train Model
```
cd src
python main.py
```

## Test Model
```
cd src
python test.py
```
