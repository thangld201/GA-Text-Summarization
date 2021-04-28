# GA-Text-Summarization

## Install Dependencies

```
pip install requirements.txt
```

## Format Dataset

This splits the dataset into a body (the actual article) and highlights (the summary). Does not split dataset into training and testing.
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
